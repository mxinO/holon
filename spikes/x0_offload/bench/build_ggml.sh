#!/usr/bin/env bash
# Build ggml-base + ggml-cpu (through llama.cpp's top-level CMake; nothing else is compiled) (static, PIC, no OpenMP) from a llama.cpp checkout for one ISA tier:
#   build_ggml.sh <llama.cpp checkout> <out dir> native|avx2
# native = -march=native (AVX-512 where the CPU has it); avx2 = AVX2/FMA/F16C/BMI2 only.
set -euo pipefail
SRC=$1 OUT=$2 ISA=$3
case $ISA in
  native) FLAGS=(-DGGML_NATIVE=ON) ;;
  avx2) FLAGS=(-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON
               -DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF -DGGML_AVX512_VNNI=OFF -DGGML_AVX512_BF16=OFF) ;;
  *) echo "isa: native or avx2" >&2; exit 2 ;;
esac
mkdir -p "$OUT"
cmake -S "$SRC" -B "$OUT" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DGGML_CUDA=OFF -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF -DGGML_CCACHE=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF "${FLAGS[@]}" > "$OUT.cmake.log"
cmake --build "$OUT" --target ggml-cpu ggml-base -j "$(nproc)" > "$OUT.build.log"
find "$OUT" -name 'libggml-cpu.a' -o -name 'libggml-base.a'
