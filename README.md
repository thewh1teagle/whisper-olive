# whisper-ort

(WIP) All in one whisper transcription. 

Features
- Single model file
- Optimized by onnxruntime (almost every GPU)
- Very simple implementation. Almost everything happens in onnx runtime

Export all in one model in https://github.com/microsoft/Olive/tree/main/examples/whisper

```console
python prepare_whisper_configs.py --model_name openai/whisper-tiny.en
python -m olive run --config whisper_cpu_int8.json --setup
python -m olive run --config whisper_cpu_int8.json
python test_transcription.py --config whisper_cpu_int8.json
```

Requires onnx runtime with enabled `--use_extensions`
Onnx build instructions:
https://onnxruntime.ai/docs/build/inferencing.html

Build onnx:
```console
git clone https://github.com/microsoft/onnxruntime.git
cd onnxruntime
./build.sh --config RelWithDebInfo --parallel --compile_no_warning_as_error --skip_submodule_sync --cmake_extra_defines CMAKE_OSX_ARCHITECTURES=arm64 --use_extensions --use_coreml
```

Build program:
```console
export ORT_LIB_PROFILE=Debug
export ORT_LIB_LOCATION=~/Documents/whisper-ort/onnxruntime/build/MacOS/RelWithDebInfo
wget https://github.com/thewh1teagle/vibe/raw/main/samples/single.wav
cargo run single.wav
```

Things to make it stable / usable:
1. Ask onnx team to enable this flag so we'll get pre built onnx lib to link static / dynamic with extensions enabled.
2. Ask ort to enable this flag
3. Build custom onnx for all platforms: `macOS Intel`, `macOS Silicon`, `Windows x86-64`, `Linux x86-64`.
-   Enable DirectML for Windows. 
-   Enable CoreML for macOS.
-   Make sure GPU model can fallback to CPU. (Ask Olive team)
-   Create Github CI to build it. Can start from csukuangfj CI example.
1. Use pyannote-rs + whisper-ort. If segment is bigger than 30s than fallabck and iterate it (sliding window).
2. Support multi language
-   Enable `--multiligual` in model creatino
-   Generate `forced_decoder_ids` from langauge id
-   Make sure it handles correctly foreign languages with special characters