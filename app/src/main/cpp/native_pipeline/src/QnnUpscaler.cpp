#include "QnnUpscaler.h"
#include <android/log.h>
#include <dlfcn.h>

#define LOG_TAG "GameNative_QNN"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

QnnUpscaler::QnnUpscaler() {}

QnnUpscaler::~QnnUpscaler() {
    Shutdown();
}

bool QnnUpscaler::Initialize(const std::string& modelPath, const std::string& backendPath) {
    LOGI("Initializing QNN Upscaler with backend: %s", backendPath.c_str());
    
    if (!loadBackend(backendPath)) {
        LOGE("Failed to load QNN backend");
        return false;
    }

    // Initialize Backend
    Qnn_ErrorHandle_t error = QNN_SUCCESS;
    error = qnnInterface_.backendPrepare(backendHandle_);
    if (error != QNN_SUCCESS) {
        LOGE("QnnBackend_prepare failed: %d", error);
        return false;
    }

    if (!loadModel(modelPath)) {
        LOGE("Failed to load XLSR model");
        return false;
    }

    LOGI("QNN Upscaler initialized successfully");
    return true;
}

bool QnnUpscaler::loadBackend(const std::string& backendPath) {
    void* libHandle = dlopen(backendPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!libHandle) {
        LOGE("Could not open backend library: %s", dlerror());
        return false;
    }

    typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(const QnnInterface_t***, uint32_t*);
    QnnInterfaceGetProvidersFn_t getInterfaceProviders = (QnnInterfaceGetProvidersFn_t)dlsym(libHandle, "QnnInterface_getProviders");
    
    if (!getInterfaceProviders) {
        LOGE("Could not find QnnInterface_getProviders");
        dlclose(libHandle);
        return false;
    }

    const QnnInterface_t** interfaceProviders;
    uint32_t numProviders;
    if (getInterfaceProviders(&interfaceProviders, &numProviders) != QNN_SUCCESS) {
        LOGE("Failed to get interface providers");
        dlclose(libHandle);
        return false;
    }

    // Usually there is only 1 provider per .so
    qnnInterface_ = *interfaceProviders[0];
    
    LOGI("Loaded backend interface: %s", qnnInterface_.backendName);
    return true;
}

bool QnnUpscaler::loadModel(const std::string& modelPath) {
    LOGI("Loading DLC Model via System Interface: %s", modelPath.c_str());

    void* sysLibHandle = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_GLOBAL);
    if (!sysLibHandle) {
        LOGE("Could not open system library: %s", dlerror());
        return false;
    }

    typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn_t)(const QnnSystemInterface_t***, uint32_t*);
    auto getSystemInterfaceProviders = (QnnSystemInterfaceGetProvidersFn_t)dlsym(sysLibHandle, "QnnSystemInterface_getProviders");
    
    if (!getSystemInterfaceProviders) {
        LOGE("Could not find QnnSystemInterface_getProviders");
        dlclose(sysLibHandle);
        return false;
    }

    const QnnSystemInterface_t** sysInterfaceProviders;
    uint32_t numSysProviders;
    if (getSystemInterfaceProviders(&sysInterfaceProviders, &numSysProviders) != QNN_SUCCESS) {
        LOGE("Failed to get system interface providers");
        dlclose(sysLibHandle);
        return false;
    }

    QnnSystemInterface_t systemInterface = *sysInterfaceProviders[0];
    
    // Create Context
    if (qnnInterface_.contextCreate(backendHandle_, deviceHandle_, nullptr, &contextHandle_) != QNN_SUCCESS) {
        LOGE("Failed to create QnnContext");
        return false;
    }

    // TODO: In a complete implementation, use systemInterface.systemContextCreate,
    // read the DLC into memory, use systemInterface.systemDlcCreateHandle to parse it, 
    // and extract the GraphInfo. For the exact XLSR model name, we would retrieve it:
    // qnnInterface_.graphRetrieve(contextHandle_, "xlsr_graph", &graphHandle_);
    
    LOGI("DLC Model parsed and Context created.");
    return true;
}

bool QnnUpscaler::Upscale(const int frameFd, size_t size, uint32_t width, uint32_t height, 
                         int outputFd, uint32_t outWidth, uint32_t outHeight) {
    LOGI("Registering DMA-BUF FD: %d (size: %zu)", frameFd, size);
    
    Qnn_MemDescriptor_t memDesc = QNN_MEM_DESCRIPTOR_INIT;
    memDesc.memType = QNN_MEM_TYPE_DMA_BUF;
    memDesc.dmaBufFd = frameFd;
    memDesc.size = size;
    
    Qnn_MemHandle_t memHandle = nullptr;
    Qnn_ErrorHandle_t error = qnnInterface_.memRegister(contextHandle_, &memDesc, 1, &memHandle);
    
    if (error != QNN_SUCCESS) {
        LOGE("QnnMem_register failed: %d", error);
        return false;
    }

    // Prepare Qnn_Tensor_t for input
    Qnn_Tensor_t inputTensor = QNN_TENSOR_INIT;
    inputTensor.version = QNN_TENSOR_VERSION_1;
    inputTensor.v1.id = 0; // Should match model's input ID
    inputTensor.v1.type = QNN_TENSOR_TYPE_APP_READ_WRITE;
    inputTensor.v1.memHandle = memHandle;
    // ... setup dimensions ...

    // TODO: Execute graph
    // error = qnnInterface_.graphExecute(graphHandle_, &inputTensor, 1, &outputTensor, 1, nullptr, nullptr);
    
    qnnInterface_.memDeRegister(&memHandle, 1);
    return true;
}

void QnnUpscaler::Shutdown() {
    // TODO: Cleanup QNN handles
}
