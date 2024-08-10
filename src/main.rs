use ndarray::{Array, Axis};
use ort::{GraphOptimizationLevel, Session};
use std::{fs::File, io::Read, time::Instant};

fn main() {
    let extension_lib_name = if cfg!(windows) {
        "ortextensions.dll"
    } else if cfg!(target_os = "macos") {
        "libortextensions.dylib"
    } else {
        "libortextensions.so"
    };

    let audio_path = std::env::args().nth(1).expect("Please specify audio file");
    let session = Session::builder()
        .unwrap()
        .with_operator_library(extension_lib_name)
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap()
        .with_intra_threads(4)
        .unwrap()
        .commit_from_file("whisper_medium_cpu_int8.onnx")
        .unwrap();

    let mut audio_file = File::open(audio_path).unwrap();
    let mut audio_buffer = Vec::new();
    audio_file.read_to_end(&mut audio_buffer).unwrap();
    let audio = ndarray::Array1::from_iter(audio_buffer);
    let audio = audio.view().insert_axis(Axis(0));

    // Hyper parameters
    let max_length = Array::from_shape_vec((1,), vec![30i32]).unwrap();
    let min_length = Array::from_shape_vec((1,), vec![1i32]).unwrap();
    let num_beams = Array::from_shape_vec((1,), vec![5i32]).unwrap();
    let num_return_sequences = Array::from_shape_vec((1,), vec![1i32]).unwrap();
    let length_penalty = Array::from_shape_vec((1,), vec![1.0f32]).unwrap();
    let repetition_penalty = Array::from_shape_vec((1,), vec![1.0f32]).unwrap();
    let decoder_input_ids =
        Array::from_shape_vec((1, 4), vec![50258, 50259, 50359, 50363]).unwrap(); // Change 50259 to 50279 for hebrew

    let inputs = ort::inputs![
        "audio_stream" => audio.view(),
        "max_length" => max_length.view(),
        "min_length" => min_length.view(),
        "num_beams" => num_beams.view(),
        "num_return_sequences" => num_return_sequences.view(),
        "length_penalty" => length_penalty.view(),
        "repetition_penalty" => repetition_penalty.view(),
        "decoder_input_ids" => decoder_input_ids.view()
    ]
    .unwrap();

    let start = Instant::now();
    let outputs = session.run(inputs).unwrap();
    let output = outputs.get("str").unwrap();

    println!(
        "Output: {:?}",
        output.try_extract_string_tensor().unwrap().first().unwrap()
    );
    println!("Took {} seconds", start.elapsed().as_secs());
}
