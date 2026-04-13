package app.gamenative.neuralbench;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.Gravity;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class NeuralBenchActivity extends Activity {
    private static final String TAG = "NeuralBench";
    private static final String MODEL_ASSET = "models/quicksrnetsmall_320x180_4x.tflite";

    private ExecutorService executor;
    private Interpreter interpreter;
    private GpuDelegate gpuDelegate;
    private String delegateLabel = "Not initialized";

    private TextView statusView;
    private Button runButton;
    private ImageView previewView;
    private boolean benchmarkStarted;

    private int inputWidth;
    private int inputHeight;
    private int outputWidth;
    private int outputHeight;
    private ByteBuffer inputTensor;
    private ByteBuffer outputTensor;
    private int[] previewPixels;
    private Bitmap previewBitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        executor = Executors.newSingleThreadExecutor();
        buildUi();
        executor.execute(() -> {
            try {
                initializeModel();
                runOnUiThread(() -> {
                    runButton.setEnabled(true);
                    statusView.setText(baseStatus() + "\n\nStarting benchmark...");
                    if (!benchmarkStarted) {
                        benchmarkStarted = true;
                        runBenchmark();
                    }
                });
            } catch (Throwable throwable) {
                Log.e(TAG, "Initialization failed", throwable);
                runOnUiThread(() -> statusView.setText("Initialization failed:\n" + throwable));
            }
        });
    }

    @Override
    protected void onDestroy() {
        if (executor != null) {
            executor.shutdownNow();
        }
        if (interpreter != null) {
            interpreter.close();
        }
        if (gpuDelegate != null) {
            gpuDelegate.close();
        }
        super.onDestroy();
    }

    private void buildUi() {
        ScrollView scrollView = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setGravity(Gravity.CENTER_HORIZONTAL);
        root.setPadding(dp(18), dp(18), dp(18), dp(18));
        scrollView.addView(root);

        TextView title = new TextView(this);
        title.setText("QuickSRNet 720p Benchmark");
        title.setTextColor(Color.WHITE);
        title.setTextSize(24f);
        title.setGravity(Gravity.CENTER);
        root.addView(title, matchWrap());

        statusView = new TextView(this);
        statusView.setText("Loading model...");
        statusView.setTextColor(Color.rgb(220, 230, 241));
        statusView.setTextSize(14f);
        statusView.setPadding(0, dp(12), 0, dp(12));
        root.addView(statusView, matchWrap());

        runButton = new Button(this);
        runButton.setText("Run Benchmark");
        runButton.setEnabled(false);
        runButton.setOnClickListener(view -> runBenchmark());
        root.addView(runButton, matchWrap());

        previewView = new ImageView(this);
        previewView.setBackgroundColor(Color.rgb(8, 11, 16));
        previewView.setScaleType(ImageView.ScaleType.FIT_CENTER);
        LinearLayout.LayoutParams previewParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                dp(420));
        previewParams.setMargins(0, dp(14), 0, 0);
        root.addView(previewView, previewParams);

        setContentView(scrollView);
    }

    private LinearLayout.LayoutParams matchWrap() {
        return new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
    }

    private void initializeModel() throws IOException {
        ByteBuffer modelBuffer = loadAsset(MODEL_ASSET);
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4);
        try {
            gpuDelegate = new GpuDelegate();
            options.addDelegate(gpuDelegate);
            interpreter = new Interpreter(modelBuffer, options);
            delegateLabel = "TensorFlow Lite GPU delegate";
        } catch (Throwable gpuError) {
            Log.w(TAG, "GPU delegate failed, falling back to CPU", gpuError);
            if (gpuDelegate != null) {
                gpuDelegate.close();
                gpuDelegate = null;
            }
            modelBuffer.rewind();
            Interpreter.Options cpuOptions = new Interpreter.Options();
            cpuOptions.setNumThreads(4);
            interpreter = new Interpreter(modelBuffer, cpuOptions);
            delegateLabel = "CPU fallback: " + gpuError.getClass().getSimpleName();
        }

        Tensor input = interpreter.getInputTensor(0);
        Tensor output = interpreter.getOutputTensor(0);
        int[] inputShape = input.shape();
        int[] outputShape = output.shape();
        if (input.dataType() != DataType.FLOAT32 || output.dataType() != DataType.FLOAT32) {
            throw new IllegalStateException("Expected FLOAT32 tensors, got input="
                    + input.dataType() + " output=" + output.dataType());
        }
        if (!isNhwcRgb(inputShape) || !isNhwcRgb(outputShape)) {
            throw new IllegalStateException("Expected NHWC RGB tensors. input="
                    + Arrays.toString(inputShape) + " output=" + Arrays.toString(outputShape));
        }

        inputHeight = inputShape[1];
        inputWidth = inputShape[2];
        outputHeight = outputShape[1];
        outputWidth = outputShape[2];

        inputTensor = ByteBuffer.allocateDirect(inputWidth * inputHeight * 3 * Float.BYTES)
                .order(ByteOrder.nativeOrder());
        outputTensor = ByteBuffer.allocateDirect(outputWidth * outputHeight * 3 * Float.BYTES)
                .order(ByteOrder.nativeOrder());
        previewPixels = new int[outputWidth * outputHeight];
        previewBitmap = Bitmap.createBitmap(outputWidth, outputHeight, Bitmap.Config.ARGB_8888);

        fillInputPattern(0);
        Log.i(TAG, baseStatus().replace('\n', ' '));
    }

    private boolean isNhwcRgb(int[] shape) {
        return shape.length == 4 && shape[0] == 1 && shape[3] == 3;
    }

    private void runBenchmark() {
        runButton.setEnabled(false);
        statusView.setText(baseStatus() + "\n\nRunning benchmarks...");
        executor.execute(() -> {
            try {
                BenchmarkReport report = runAllBenchmarks();
                Log.i(TAG, "\n" + report.text);
                runOnUiThread(() -> {
                    statusView.setText(report.text);
                    previewView.setImageBitmap(previewBitmap);
                    runButton.setEnabled(true);
                });
            } catch (Throwable throwable) {
                Log.e(TAG, "Benchmark failed", throwable);
                runOnUiThread(() -> {
                    statusView.setText("Benchmark failed:\n" + throwable);
                    runButton.setEnabled(true);
                });
            }
        });
    }

    private BenchmarkReport runAllBenchmarks() {
        fillInputPattern(0);
        runInferenceLoops(20, false, 0);

        TimedResult pureInference = runInferenceLoops(240, false, 0);
        TimedResult inputAndInference = runInferenceLoops(180, true, 0);
        TimedResult occasionalPreview = runInferenceLoops(180, true, 30);
        TimedResult everyFramePreview = runInferenceLoops(45, true, 1);

        tensorToPreviewBitmap();

        String text = baseStatus()
                + "\n\nBenchmark results:"
                + "\nPure TFLite run only: " + pureInference.summary()
                + "\nGenerate input + run: " + inputAndInference.summary()
                + "\nPreview every 30 frames: " + occasionalPreview.summary()
                + "\nPreview every frame: " + everyFramePreview.summary()
                + "\n\nInterpretation:"
                + "\nPure run includes TFLite synchronization into the output buffer."
                + "\nPreview modes include CPU float-to-bitmap conversion and are intentionally bad-path measurements."
                + "\nA renderer integration must avoid the preview path entirely.";
        return new BenchmarkReport(text);
    }

    private TimedResult runInferenceLoops(int frames, boolean regenerateInput, int previewEvery) {
        long totalStart = System.nanoTime();
        double inputMs = 0.0;
        double runMs = 0.0;
        double previewMs = 0.0;
        int previewCount = 0;

        for (int frame = 0; frame < frames; frame++) {
            if (regenerateInput) {
                long inputStart = System.nanoTime();
                fillInputPattern(frame);
                inputMs += elapsedMs(inputStart);
            }

            long runStart = System.nanoTime();
            outputTensor.rewind();
            interpreter.run(inputTensor, outputTensor);
            runMs += elapsedMs(runStart);

            if (previewEvery > 0 && frame % previewEvery == 0) {
                long previewStart = System.nanoTime();
                tensorToPreviewBitmap();
                previewMs += elapsedMs(previewStart);
                previewCount++;
            }
        }

        return new TimedResult(frames, previewCount, elapsedMs(totalStart), inputMs, runMs, previewMs);
    }

    private void fillInputPattern(int frame) {
        inputTensor.rewind();
        float t = (frame % 240) / 240f;
        float movingX = 0.12f + 0.76f * t;
        for (int y = 0; y < inputHeight; y++) {
            float ny = y / (float) Math.max(1, inputHeight - 1);
            for (int x = 0; x < inputWidth; x++) {
                float nx = x / (float) Math.max(1, inputWidth - 1);
                float grid = ((x / 12 + y / 12) & 1) == 0 ? 0.08f : 0.18f;
                float dx = nx - movingX;
                float dy = ny - (0.52f + 0.18f * (float) Math.sin(frame * 0.05f));
                float ball = dx * dx + dy * dy < 0.006f ? 1.0f : 0.0f;
                float block = nx > 0.12f && nx < 0.36f && ny > 0.18f && ny < 0.42f ? 1.0f : 0.0f;
                float red = clamp01(grid + ball * 1.0f + block * 0.05f);
                float green = clamp01(grid + ball * 0.78f + block * 0.85f);
                float blue = clamp01(grid + ball * 0.20f + block * 1.0f);
                inputTensor.putFloat(red);
                inputTensor.putFloat(green);
                inputTensor.putFloat(blue);
            }
        }
        inputTensor.rewind();
    }

    private void tensorToPreviewBitmap() {
        outputTensor.rewind();
        for (int y = 0; y < outputHeight; y++) {
            for (int x = 0; x < outputWidth; x++) {
                previewPixels[y * outputWidth + x] = Color.rgb(
                        toByte(outputTensor.getFloat()),
                        toByte(outputTensor.getFloat()),
                        toByte(outputTensor.getFloat()));
            }
        }
        outputTensor.rewind();
        previewBitmap.setPixels(previewPixels, 0, outputWidth, 0, 0, outputWidth, outputHeight);
    }

    private ByteBuffer loadAsset(String assetPath) throws IOException {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        byte[] buffer = new byte[16 * 1024];
        try (InputStream input = getAssets().open(assetPath)) {
            int read;
            while ((read = input.read(buffer)) != -1) {
                output.write(buffer, 0, read);
            }
        }
        byte[] bytes = output.toByteArray();
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(bytes.length);
        byteBuffer.order(ByteOrder.nativeOrder());
        byteBuffer.put(bytes);
        byteBuffer.rewind();
        return byteBuffer;
    }

    private String baseStatus() {
        return "Model: " + MODEL_ASSET
                + "\nDelegate: " + delegateLabel
                + "\nInput: " + inputWidth + "x" + inputHeight
                + "\nOutput: " + outputWidth + "x" + outputHeight;
    }

    private int toByte(float value) {
        if (Float.isNaN(value)) {
            return 0;
        }
        return Math.max(0, Math.min(255, Math.round(value * 255f)));
    }

    private float clamp01(float value) {
        return Math.max(0f, Math.min(1f, value));
    }

    private double elapsedMs(long startNanos) {
        return (System.nanoTime() - startNanos) / 1_000_000.0;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private static final class BenchmarkReport {
        final String text;

        BenchmarkReport(String text) {
            this.text = text;
        }
    }

    private static final class TimedResult {
        final int frames;
        final int previewCount;
        final double totalMs;
        final double inputMs;
        final double runMs;
        final double previewMs;

        TimedResult(int frames, int previewCount, double totalMs, double inputMs, double runMs, double previewMs) {
            this.frames = frames;
            this.previewCount = previewCount;
            this.totalMs = totalMs;
            this.inputMs = inputMs;
            this.runMs = runMs;
            this.previewMs = previewMs;
        }

        String summary() {
            double runAvg = runMs / Math.max(1, frames);
            double totalAvg = totalMs / Math.max(1, frames);
            double fps = 1000.0 / Math.max(0.001, totalAvg);
            double inputAvg = inputMs / Math.max(1, frames);
            double previewAvg = previewCount == 0 ? 0.0 : previewMs / previewCount;
            return String.format(
                    Locale.US,
                    "run %.2f ms, total %.2f ms/frame, %.1f FPS, input %.2f ms/frame, preview %.2f ms/preview (%d previews)",
                    runAvg,
                    totalAvg,
                    fps,
                    inputAvg,
                    previewAvg,
                    previewCount);
        }
    }
}
