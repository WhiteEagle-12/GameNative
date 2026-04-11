#include "BufferInterceptor.h"
#include <android/log.h>
#include <vndk/hardware_buffer.h>

#define LOG_TAG "GameNative_Buffer"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Forward declaration of native_handle structure if not available in standard headers
typedef struct native_handle {
    int version;        /* sizeof(native_handle_t) */
    int numFds;         /* number of file-descriptors at &data[0] */
    int numInts;        /* number of ints at &data[numFds] */
    int data[0];        /* numFds + numInts ints */
} native_handle_t;

BufferInterceptor::BufferInterceptor() {}

BufferInterceptor::~BufferInterceptor() {
    Shutdown();
}

bool BufferInterceptor::Initialize() {
    LOGI("BufferInterceptor Initialized.");
    return true;
}

int BufferInterceptor::GetDmaBufFd(JNIEnv* env, jobject hardwareBufferObj, size_t* outSize) {
    if (!hardwareBufferObj) {
        LOGE("Provided HardwareBuffer is null");
        return -1;
    }

    AHardwareBuffer* buffer = AHardwareBuffer_fromHardwareBuffer(env, hardwareBufferObj);
    if (!buffer) {
        LOGE("Failed to get AHardwareBuffer from jobject");
        return -1;
    }

    AHardwareBuffer_Desc desc;
    AHardwareBuffer_describe(buffer, &desc);
    
    // Estimate size (width * height * bpp + alignment) - QNN usually requires accurate allocated size
    // We roughly estimate based on format. For precise DMA-BUF size we might need lseek or ioctl.
    uint32_t bpp = 4; // Assuming RGBA_8888
    if (desc.format == AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM) bpp = 4;
    else if (desc.format == AHARDWAREBUFFER_FORMAT_R8G8B8_UNORM) bpp = 3;
    else if (desc.format == AHARDWAREBUFFER_FORMAT_R5G6B5_UNORM) bpp = 2;

    *outSize = desc.stride * desc.height * bpp;
    
    const native_handle_t* handle = AHardwareBuffer_getNativeHandle(buffer);
    if (!handle || handle->numFds < 1) {
        LOGE("HardwareBuffer does not have a valid native_handle or FD");
        // AHardwareBuffer_release(buffer); // Release logic moved to client or wrapper
        return -1;
    }

    int dmaBufFd = handle->data[0];
    LOGI("Successfully extracted DMA-BUF FD: %d, Size: %zu (WxH: %dx%d, Stride: %d)", 
         dmaBufFd, *outSize, desc.width, desc.height, desc.stride);

    return dmaBufFd;
}

void BufferInterceptor::ReleaseBuffer(AHardwareBuffer* buffer) {
    if (buffer) {
        AHardwareBuffer_release(buffer);
    }
}

void BufferInterceptor::Shutdown() {
    LOGI("BufferInterceptor Shutdown.");
}
