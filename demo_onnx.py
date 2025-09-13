import os
import argparse
import numpy as np
import onnxruntime as ort
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import json
import time
from tqdm import tqdm
import cv2


class ONNXImageClassifier:
    """ONNX Image Classification Inference Class"""
    
    def __init__(self, model_path, metadata_path=None, providers=None):
        """
        Initialize ONNX model for inference
        
        Args:
            model_path: Path to ONNX model
            metadata_path: Path to model metadata JSON
            providers: ONNX Runtime providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        """
        # Set providers
        if providers is None:
            # Auto-detect available providers
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                self.providers = ['CPUExecutionProvider']
        else:
            self.providers = providers
        
        print(f"Using providers: {self.providers}")
        
        # Load ONNX model
        self.session = ort.InferenceSession(model_path, providers=self.providers)
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        
        # Load metadata if available
        self.metadata = None
        self.class_names = None
        self.input_size = 224
        
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                self.class_names = self.metadata.get('class_names', None)
                self.input_size = self.metadata.get('input_size', 224)
                print(f"Loaded metadata from {metadata_path}")
        
        # Set default input size from model if not in metadata
        if len(self.input_shape) == 4 and self.input_shape[2] > 0:
            self.input_size = self.input_shape[2]
        
        print(f"Model loaded successfully!")
        print(f"Input shape: {self.input_shape}")
        print(f"Input size: {self.input_size}")
    
    def preprocess(self, image):
        """Preprocess image for inference"""
        if isinstance(image, str):
            # Load image from path
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            image = Image.fromarray(image)
        
        # Apply transforms
        transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        tensor = transform(image)
        # Add batch dimension
        tensor = tensor.unsqueeze(0).numpy()
        
        return tensor, image
    
    def predict(self, image, top_k=5):
        """
        Run inference on image
        
        Args:
            image: Image path, PIL Image, or numpy array
            top_k: Number of top predictions to return
        
        Returns:
            predictions: List of (class_idx, class_name, probability) tuples
        """
        # Preprocess image
        input_tensor, original_image = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        
        # Apply softmax to get probabilities
        probabilities = self._softmax(outputs[0])
        
        # Get top k predictions
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        
        predictions = []
        for idx in top_indices:
            class_name = self.class_names[idx] if self.class_names else f"Class {idx}"
            predictions.append((idx, class_name, probabilities[idx]))
        
        return predictions, original_image
    
    def _softmax(self, x):
        """Compute softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def benchmark(self, num_runs=100, warmup=10):
        """Benchmark inference speed"""
        print(f"\nBenchmarking inference speed...")
        
        # Create dummy input
        dummy_input = np.random.randn(1, 3, self.input_size, self.input_size).astype(np.float32)
        
        # Warmup
        print(f"Warming up with {warmup} runs...")
        for _ in range(warmup):
            _ = self.session.run([self.output_name], {self.input_name: dummy_input})
        
        # Benchmark
        print(f"Running {num_runs} inferences...")
        times = []
        for _ in tqdm(range(num_runs)):
            start = time.perf_counter()
            _ = self.session.run([self.output_name], {self.input_name: dummy_input})
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        times = np.array(times)
        
        print(f"\nBenchmark Results:")
        print(f"  Average: {np.mean(times):.2f} ms")
        print(f"  Median:  {np.median(times):.2f} ms")
        print(f"  Min:     {np.min(times):.2f} ms")
        print(f"  Max:     {np.max(times):.2f} ms")
        print(f"  Std:     {np.std(times):.2f} ms")
        print(f"  FPS:     {1000/np.mean(times):.1f}")
        
        return times


def visualize_predictions(image, predictions, save_path=None):
    """Visualize predictions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display image
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('Input Image')
    
    # Display predictions
    classes = [p[1] for p in predictions]
    probs = [p[2] for p in predictions]
    y_pos = np.arange(len(classes))
    
    ax2.barh(y_pos, probs)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(classes)
    ax2.invert_yaxis()
    ax2.set_xlabel('Probability')
    ax2.set_title('Top Predictions')
    
    # Add probability values
    for i, prob in enumerate(probs):
        ax2.text(prob, i, f'{prob:.2%}', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def process_video(classifier, video_path, output_path=None, skip_frames=5):
    """Process video file for classification"""
    cap = cv2.VideoCapture(video_path)
    
    if output_path:
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    predictions_history = []
    
    print(f"Processing video: {video_path}")
    
    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            pbar.update(1)
            
            # Process every nth frame
            if frame_count % skip_frames == 0:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Get predictions
                predictions, _ = classifier.predict(rgb_frame, top_k=1)
                top_class = predictions[0][1]
                top_prob = predictions[0][2]
                
                predictions_history.append({
                    'frame': frame_count,
                    'class': top_class,
                    'probability': float(top_prob)
                })
                
                # Draw predictions on frame
                text = f"{top_class}: {top_prob:.2%}"
                cv2.putText(frame, text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if output_path:
                out.write(frame)
            
            frame_count += 1
    
    cap.release()
    if output_path:
        out.release()
        print(f"Output video saved to {output_path}")
    
    return predictions_history


def process_batch(classifier, image_dir, output_file=None):
    """Process batch of images"""
    # Get all image files
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(extensions)]
    
    print(f"Processing {len(image_files)} images from {image_dir}")
    
    results = []
    for img_file in tqdm(image_files):
        img_path = os.path.join(image_dir, img_file)
        predictions, _ = classifier.predict(img_path, top_k=1)
        
        result = {
            'filename': img_file,
            'prediction': predictions[0][1],
            'confidence': float(predictions[0][2]),
            'top_5': [(p[1], float(p[2])) for p in predictions[:5]]
        }
        results.append(result)
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    
    # Print summary
    print("\nClassification Summary:")
    class_counts = {}
    for r in results:
        cls = r['prediction']
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls}: {count} images")
    
    return results


def main(args):
    # Load model
    print(f"Loading ONNX model from {args.model_path}")
    
    # Auto-detect metadata file if not provided
    if not args.metadata_path:
        model_dir = os.path.dirname(args.model_path) or '.'
        metadata_path = os.path.join(model_dir, 'model_metadata.json')
        if os.path.exists(metadata_path):
            args.metadata_path = metadata_path
            print(f"Auto-detected metadata file: {metadata_path}")
    
    # Set providers
    providers = None
    if args.use_gpu:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    elif args.use_cpu:
        providers = ['CPUExecutionProvider']
    
    # Initialize classifier
    classifier = ONNXImageClassifier(
        args.model_path, 
        args.metadata_path,
        providers=providers
    )
    
    # Load class names if provided separately
    if args.class_names_file and os.path.exists(args.class_names_file):
        with open(args.class_names_file, 'r') as f:
            classifier.class_names = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(classifier.class_names)} class names")
    
    # Run benchmark if requested
    if args.benchmark:
        classifier.benchmark(num_runs=args.benchmark_runs)
        return
    
    # Process based on input type
    if args.video_path:
        # Process video
        output_path = args.output_path or 'output_video.mp4'
        predictions = process_video(
            classifier, 
            args.video_path, 
            output_path,
            skip_frames=args.skip_frames
        )
        
        # Save predictions history
        if args.save_predictions:
            pred_file = output_path.replace('.mp4', '_predictions.json')
            with open(pred_file, 'w') as f:
                json.dump(predictions, f, indent=2)
            print(f"Predictions saved to {pred_file}")
    
    elif args.batch_dir:
        # Process batch of images
        output_file = args.output_path or 'batch_results.json'
        results = process_batch(classifier, args.batch_dir, output_file)
        
        # Print statistics
        if results:
            confidences = [r['confidence'] for r in results]
            print(f"\nConfidence Statistics:")
            print(f"  Mean: {np.mean(confidences):.2%}")
            print(f"  Std:  {np.std(confidences):.2%}")
            print(f"  Min:  {np.min(confidences):.2%}")
            print(f"  Max:  {np.max(confidences):.2%}")
    
    elif args.image_path:
        # Process single image
        if os.path.isfile(args.image_path):
            # Single image
            print(f"\nProcessing image: {args.image_path}")
            predictions, image = classifier.predict(args.image_path, top_k=args.top_k)
            
            # Print predictions
            print("\nTop Predictions:")
            for i, (idx, class_name, prob) in enumerate(predictions, 1):
                print(f"{i}. {class_name}: {prob:.2%}")
            
            # Visualize if requested
            if args.visualize:
                save_path = args.output_path or None
                visualize_predictions(image, predictions, save_path)
            
            # Save results
            if args.save_predictions:
                results = {
                    'image': args.image_path,
                    'predictions': [
                        {'class': p[1], 'confidence': float(p[2])} 
                        for p in predictions
                    ]
                }
                output_file = args.output_path or 'predictions.json'
                if not output_file.endswith('.json'):
                    output_file = output_file.replace('.png', '.json').replace('.jpg', '.json')
                
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to {output_file}")
        
        elif os.path.isdir(args.image_path):
            # Directory of images
            results = process_batch(classifier, args.image_path, args.output_path)
    
    else:
        print("Please provide an image path, video path, or batch directory")
        return
    
    print("\n✅ Processing completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ONNX model inference for image classification')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to ONNX model')
    parser.add_argument('--metadata_path', type=str,
                       help='Path to model metadata JSON (auto-detected if not provided)')
    parser.add_argument('--class_names_file', type=str,
                       help='Path to class names file')
    
    # Input arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image_path', type=str,
                            help='Path to input image or directory')
    input_group.add_argument('--video_path', type=str,
                            help='Path to input video')
    input_group.add_argument('--batch_dir', type=str,
                            help='Directory containing images for batch processing')
    input_group.add_argument('--benchmark', action='store_true',
                            help='Run benchmark only')
    
    # Output arguments
    parser.add_argument('--output_path', type=str,
                       help='Output path for results')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save predictions to JSON file')
    
    # Inference arguments
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top predictions to show')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize predictions (for single image)')
    
    # Video processing arguments
    parser.add_argument('--skip_frames', type=int, default=5,
                       help='Process every nth frame in video')
    
    # Performance arguments
    parser.add_argument('--use_gpu', action='store_true',
                       help='Use GPU for inference (CUDA)')
    parser.add_argument('--use_cpu', action='store_true',
                       help='Force CPU inference')
    parser.add_argument('--benchmark_runs', type=int, default=100,
                       help='Number of runs for benchmarking')
    
    args = parser.parse_args()
    main(args)
    