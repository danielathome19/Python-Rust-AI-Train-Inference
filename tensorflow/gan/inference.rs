// TODO: redo from scratch to generate an image from the generator model?
use image::GrayImage;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use python_rust_ai::{create_onnx_session, predict_outputs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let session = create_onnx_session("models/tf_gan_generator_model.onnx")?;

    // Generate a random noise vector
    let latent_dim = 100;
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();
    let input_data: Vec<f32> = (0..latent_dim).map(|_| normal.sample(&mut rng) as f32).collect::<Vec<f32>>();
    let outputs = predict_outputs(&session, input_data, &[1, latent_dim])?;
    
    for output in outputs {
        let output_array = output.try_extract::<f32>()?;
        let mut generated_image = output_array.view().to_owned().into_raw_vec();
        generated_image.iter_mut().for_each(|x| *x = (*x + 1.0) / 2.0);

        // Convert f32 values (0.0 to 1.0) to u8 values (0 to 255)
        let image_data: Vec<u8> = generated_image.iter().map(|&x| (x * 255.0) as u8).collect();

        // Save the generated image
        let image = GrayImage::from_raw(28, 28, image_data).unwrap();
        image.save("data/generated_image.png")?;
        println!("Image generation complete!");
    }

    Ok(())
}