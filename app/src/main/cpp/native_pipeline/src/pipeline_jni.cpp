#include <jni.h>
#include <string>
#include <memory>
#include "QnnUpscaler.h"
#include "AfmeManager.h"
#include "BufferInterceptor.h"

// Global instances for the pipeline
static std::unique_ptr<QnnUpscaler> g_qnnUpscaler;
static std::unique_ptr<AfmeManager> g_afmeManager;
static std::unique_ptr<BufferInterceptor> g_bufferInterceptor;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_gamenative_pipeline_UpscaleService_initPipeline(JNIEnv* env, jobject /* this */, jstring modelPath, jstring backendPath) {
    g_qnnUpscaler = std::make_unique<QnnUpscaler>();
    g_afmeManager = std::make_unique<AfmeManager>();
    g_bufferInterceptor = std::make_unique<BufferInterceptor>();

    const char* nativeModelPath = env->GetStringUTFChars(modelPath, nullptr);
    const char* nativeBackendPath = env->GetStringUTFChars(backendPath, nullptr);

    bool success = g_qnnUpscaler->Initialize(nativeModelPath, nativeBackendPath);
    if (success) {
        success = g_afmeManager->Initialize();
    }
    if (success) {
        success = g_bufferInterceptor->Initialize();
    }

    env->ReleaseStringUTFChars(modelPath, nativeModelPath);
    env->ReleaseStringUTFChars(backendPath, nativeBackendPath);

    return success ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_gamenative_pipeline_UpscaleService_processFrame(JNIEnv* env, jobject /* this */, jobject inBuffer, jobject outBuffer) {
    if (!g_qnnUpscaler || !g_afmeManager || !g_bufferInterceptor) {
        return JNI_FALSE;
    }

    size_t inSize = 0;
    int inFd = g_bufferInterceptor->GetDmaBufFd(env, inBuffer, &inSize);
    
    size_t outSize = 0;
    int outFd = g_bufferInterceptor->GetDmaBufFd(env, outBuffer, &outSize);

    if (inFd < 0 || outFd < 0) {
        return JNI_FALSE;
    }

    // Assumptions: widths and heights are queried or passed elsewhere. For simplicity, assume 1080p->4k
    // In a full implementation, these would be extracted from the HardwareBuffer descriptions.
    bool success = g_qnnUpscaler->Upscale(inFd, inSize, 1920, 1080, outFd, 3840, 2160);
    
    if (success) {
        // Run hardware frame gen / AFME immediately after upscaling
        success = g_afmeManager->ProcessFrame(outFd, outFd, 3840, 2160);
    }

    return success ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_gamenative_pipeline_UpscaleService_shutdownPipeline(JNIEnv* env, jobject /* this */) {
    g_qnnUpscaler.reset();
    g_afmeManager.reset();
    g_bufferInterceptor.reset();
}
