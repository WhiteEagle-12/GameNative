#ifndef QNN_UPSCALER_H
#define QNN_UPSCALER_H

#include <string>
#include <vector>
#include <memory>

#include "QnnInterface.h"
#include "QnnTypes.h"
#include "QnnCommon.h"
#include "QnnBackend.h"
#include "QnnDevice.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnMem.h"
#include "System/QnnSystemInterface.h"

class QnnUpscaler {
public:
    QnnUpscaler();
    ~QnnUpscaler();

    bool Initialize(const std::string& modelPath, const std::string& backendPath);
    bool Upscale(const int frameFd, size_t size, uint32_t width, uint32_t height, 
                 int outputFd, uint32_t outWidth, uint32_t outHeight);
    void Shutdown();

private:
    Qnn_BackendHandle_t backendHandle_ = nullptr;
    Qnn_DeviceHandle_t deviceHandle_ = nullptr;
    Qnn_ContextHandle_t contextHandle_ = nullptr;
    Qnn_GraphHandle_t graphHandle_ = nullptr;
    QnnInterface_t qnnInterface_;
    void* systemInterfaceProvider_ = nullptr; 

    bool loadBackend(const std::string& backendPath);
    bool loadModel(const std::string& modelPath);
};

#endif // QNN_UPSCALER_H
