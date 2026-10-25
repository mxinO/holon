// GLM step 0, N3 (docs/v1-glm.md §8, D8f): the SSD read path's rate on the bar node's NVMe, before the store is built.
//
// Reads "experts" from the model's GGUF shards with O_DIRECT: an expert is three extents (Maya-S: gate 2,162,688 B,
// up 2,162,688 B, down 2,686,976 B = 7,012,352 B), each at a random 4 KiB-aligned offset, split into `chunk`-byte
// reads. A "call" is k experts read at once (a layer's SSD misses); the time is from the first submission to the last
// completion. Modes:
//   uring  one thread, one io_uring (raw syscalls), every chunk of the call submitted at once up to `depth` in flight
//   pread  one thread, the chunks read one after another (today's per-thread path, depth 1)
//   pool   `depth` threads, each taking chunks from the call (Maya's 8 workers; a comparison, not a candidate)
// Prints one line per (mode, depth, chunk, k): ms per expert (median, p90) and GB/s. Run with the page cache evicted
// (O_DIRECT bypasses it; the check is that a buffered read of the same range is not faster).
//
//   cc -O2 -o n3_ssd spikes/glm0/n3_ssd.c -lpthread
//   ./n3_ssd <seconds-per-config> <shard> [<shard>...]      # also prints whether io_uring is available
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <linux/io_uring.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

static const size_t EXTENT[3] = {2162688, 2162688, 2686976};
#define EXPERT_BYTES (2162688 + 2162688 + 2686976)

static int nfiles;
static int fds[8];
static off_t sizes[8];

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

typedef struct { int fd; off_t off; size_t len; char* buf; } Req;

// k experts' chunk list, into `buf` (k * EXPERT_BYTES, 4 KiB aligned)
static int plan(Req* r, int k, size_t chunk, char* buf, unsigned* seed) {
    int n = 0;
    for (int e = 0; e < k; ++e)
        for (int x = 0; x < 3; ++x) {
            int f = rand_r(seed) % nfiles;
            off_t span = (sizes[f] - (off_t) EXTENT[x]) / 4096;
            off_t base = ((off_t) (((uint64_t) rand_r(seed) << 31 | rand_r(seed)) % span)) * 4096;
            for (size_t o = 0; o < EXTENT[x]; o += chunk) {
                size_t len = EXTENT[x] - o < chunk ? EXTENT[x] - o : chunk;
                r[n++] = (Req){fds[f], base + (off_t) o, len, buf};
                buf += len;
            }
        }
    return n;
}

// ---- io_uring, raw syscalls ----
typedef struct {
    int fd;
    unsigned *sq_head, *sq_tail, *sq_mask, *sq_array, *cq_head, *cq_tail, *cq_mask;
    struct io_uring_sqe* sqes;
    struct io_uring_cqe* cqes;
} Ring;

static int ring_init(Ring* R, unsigned entries) {
    struct io_uring_params p;
    memset(&p, 0, sizeof p);
    R->fd = (int) syscall(__NR_io_uring_setup, entries, &p);
    if (R->fd < 0) return -errno;
    size_t sq_sz = p.sq_off.array + p.sq_entries * sizeof(unsigned);
    size_t cq_sz = p.cq_off.cqes + p.cq_entries * sizeof(struct io_uring_cqe);
    char* sq = mmap(0, sq_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, R->fd, IORING_OFF_SQ_RING);
    char* cq = mmap(0, cq_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, R->fd, IORING_OFF_CQ_RING);
    R->sqes = mmap(0, p.sq_entries * sizeof(struct io_uring_sqe), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                   R->fd, IORING_OFF_SQES);
    if (sq == MAP_FAILED || cq == MAP_FAILED || R->sqes == MAP_FAILED) return -ENOMEM;
    R->sq_head = (unsigned*) (sq + p.sq_off.head);
    R->sq_tail = (unsigned*) (sq + p.sq_off.tail);
    R->sq_mask = (unsigned*) (sq + p.sq_off.ring_mask);
    R->sq_array = (unsigned*) (sq + p.sq_off.array);
    R->cq_head = (unsigned*) (cq + p.cq_off.head);
    R->cq_tail = (unsigned*) (cq + p.cq_off.tail);
    R->cq_mask = (unsigned*) (cq + p.cq_off.ring_mask);
    R->cqes = (struct io_uring_cqe*) (cq + p.cq_off.cqes);
    return 0;
}

static void ring_push(Ring* R, const Req* q, unsigned long tag) {
    unsigned tail = *R->sq_tail, i = tail & *R->sq_mask;
    struct io_uring_sqe* s = &R->sqes[i];
    memset(s, 0, sizeof *s);
    s->opcode = IORING_OP_READ;
    s->fd = q->fd;
    s->addr = (unsigned long) q->buf;
    s->len = (unsigned) q->len;
    s->off = (unsigned long long) q->off;
    s->user_data = tag;
    R->sq_array[i] = i;
    __atomic_store_n(R->sq_tail, tail + 1, __ATOMIC_RELEASE);
}

static double run_uring(Ring* R, Req* r, int n, int depth) {
    double t0 = now();
    int sent = 0, done = 0, inflight = 0;
    while (done < n) {
        int add = 0;
        while (sent < n && inflight < depth) { ring_push(R, &r[sent], (unsigned long) sent); ++sent; ++inflight; ++add; }
        if (syscall(__NR_io_uring_enter, R->fd, add, 1, IORING_ENTER_GETEVENTS, NULL, 0) < 0) { perror("enter"); exit(1); }
        unsigned head = *R->cq_head;
        while (head != __atomic_load_n(R->cq_tail, __ATOMIC_ACQUIRE)) {
            struct io_uring_cqe* c = &R->cqes[head & *R->cq_mask];
            if (c->res != (int) r[c->user_data].len) { fprintf(stderr, "short read %d\n", c->res); exit(1); }
            ++head; ++done; --inflight;
        }
        __atomic_store_n(R->cq_head, head, __ATOMIC_RELEASE);
    }
    return now() - t0;
}

static double run_pread(Req* r, int n) {
    double t0 = now();
    for (int i = 0; i < n; ++i)
        if (pread(r[i].fd, r[i].buf, r[i].len, r[i].off) != (ssize_t) r[i].len) { perror("pread"); exit(1); }
    return now() - t0;
}

typedef struct { Req* r; int n; int* next; } Pool;
static void* pool_worker(void* a) {
    Pool* p = a;
    for (;;) {
        int i = __atomic_fetch_add(p->next, 1, __ATOMIC_RELAXED);
        if (i >= p->n) return NULL;
        if (pread(p->r[i].fd, p->r[i].buf, p->r[i].len, p->r[i].off) != (ssize_t) p->r[i].len) { perror("pread"); exit(1); }
    }
}
static double run_pool(Req* r, int n, int threads) {
    pthread_t th[64];
    int next = 0;
    Pool p = {r, n, &next};
    double t0 = now();
    for (int t = 0; t < threads; ++t) pthread_create(&th[t], NULL, pool_worker, &p);
    for (int t = 0; t < threads; ++t) pthread_join(th[t], NULL);
    return now() - t0;
}

static int cmp(const void* a, const void* b) { double x = *(double*) a, y = *(double*) b; return (x > y) - (x < y); }

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: n3_ssd <seconds> <shard>...\n"); return 2; }
    double secs = atof(argv[1]);
    for (int i = 2; i < argc && nfiles < 8; ++i) {
        fds[nfiles] = open(argv[i], O_RDONLY | O_DIRECT);
        if (fds[nfiles] < 0) { perror(argv[i]); return 1; }
        struct stat st;
        fstat(fds[nfiles], &st);
        if (st.st_size > 64 << 20) sizes[nfiles++] = st.st_size;
    }
    Ring R = {0};
    int ur = ring_init(&R, 256);
    printf("io_uring: %s\n", ur == 0 ? "available" : strerror(-ur));
    const int KMAX = 8;
    char* buf;
    if (posix_memalign((void**) &buf, 4096, (size_t) KMAX * EXPERT_BYTES)) return 1;
    Req* r = malloc(sizeof(Req) * KMAX * 3 * (EXPERT_BYTES / 4096 + 3));
    unsigned seed = 7;
    const char* modes[] = {"uring", "pread", "pool"};
    const int depths[] = {4, 8, 16, 32};
    const size_t chunks[] = {131072, 262144, 524288, 1048576};
    const int ks[] = {1, 2, 4, 8};
    printf("mode depth chunk_kib k calls ms_per_expert_p50 ms_per_expert_p90 GBps\n");
    for (int m = 0; m < 3; ++m) {
        if (m == 0 && ur != 0) continue;
        for (int d = 0; d < 4; ++d) {
            if (m == 1 && d > 0) continue;                  // pread: depth 1
            for (int c = 0; c < 4; ++c)
                for (int ki = 0; ki < 4; ++ki) {
                    int k = ks[ki];
                    double t[4096], total = 0, bytes = 0;
                    int calls = 0;
                    double end = now() + secs;
                    while (now() < end && calls < 4096) {
                        int n = plan(r, k, chunks[c], buf, &seed);
                        double dt = m == 0 ? run_uring(&R, r, n, depths[d]) : m == 1 ? run_pread(r, n)
                                                                                     : run_pool(r, n, depths[d]);
                        t[calls++] = dt / k;
                        total += dt;
                        bytes += (double) k * EXPERT_BYTES;
                    }
                    qsort(t, calls, sizeof(double), cmp);
                    printf("%s %d %zu %d %d %.3f %.3f %.2f\n", modes[m], m == 1 ? 1 : depths[d], chunks[c] / 1024, k,
                           calls, t[calls / 2] * 1e3, t[calls * 9 / 10] * 1e3, bytes / total / 1e9);
                    fflush(stdout);
                }
        }
    }
    return 0;
}
