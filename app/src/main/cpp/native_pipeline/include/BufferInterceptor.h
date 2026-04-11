#ifndef BUFFER_INTERCEPTOR_H
#define BUFFER_INTERCEPTOR_H

#include <android/hardware_buffer.h>
#include <android/hardware_buffer_jni.h>
#include <jni.h>

class BufferInterceptor {
public:
    BufferInterceptor();
    ~BufferInterceptor();

    // Start interception or provide an external buffer
    bool Initialize();
    
    // Process an incoming Java HardwareBuffer object to get its DMA-BUF
    int GetDmaBufFd(JNIEnv* env, jobject hardwareBuffer, size_t* outSize);

    void ReleaseBuffer(AHardwareBuffer* buffer);
    void Shutdown();

private:
    // Any necessary EGL context or surface details could go here
};

#endif // BUFFER_INTERCEPTOR_H
