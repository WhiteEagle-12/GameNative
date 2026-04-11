#ifndef AFME_MANAGER_H
#define AFME_MANAGER_H

#include <cstdint>
#include <string>

class AfmeManager {
public:
    AfmeManager();
    ~AfmeManager();

    bool Initialize();
    bool ProcessFrame(int inputFd, int outputFd, uint32_t width, uint32_t height);
    void Shutdown();

private:
    // Snapdragon Game AI SDK specific handles will go here
    void* afmeHandle_ = nullptr;
};

#endif // AFME_MANAGER_H
