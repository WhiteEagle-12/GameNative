package com.gamenative.pipeline

import android.hardware.HardwareBuffer
import android.util.Log

class UpscaleService {

    companion object {
        private const val TAG = "GameNative_UpscaleSvc"

        init {
            try {
                System.loadLibrary("native_pipeline")
                Log.i(TAG, "Successfully loaded native_pipeline library")
            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load native_pipeline library", e)
            }
        }
    }

    /**
     * Initializes the QNN and AFME pipeline logic natively.
     * @param modelPath Path to the DLC model.
     * @param backendPath Path to the QNN backend library.
     */
    external fun initPipeline(modelPath: String, backendPath: String): Boolean

    /**
     * Processes an incoming HardwareBuffer via the NPU and outputs an upscaled buffer.
     * @param inBuffer The low-resolution emulator output buffer.
     * @param outBuffer The high-resolution display buffer to hold the result.
     */
    external fun processFrame(inBuffer: HardwareBuffer, outBuffer: HardwareBuffer): Boolean

    /**
     * Cleans up the native resources.
     */
    external fun shutdownPipeline()
}
