#pragma once

#include <string>
#include <vector>

// Tiny wrapper around ONNX Runtime. Compiles to a no-op if HAS_ORT is not set.
class OnnxRunner {
public:
    // Construct with model path. If empty, attempts to auto-locate assets/model.onnx.
    explicit OnnxRunner(const std::string& modelPath = "");
    ~OnnxRunner();

    bool ready() const;          // true if session initialized
    bool usePrev() const;        // whether model expects prev_vib
    bool hasAudio() const;       // whether model outputs audio
    int  fsVib() const;          // vib sample rate
    int  fsAud() const;          // audio sample rate (if any)

    // Run one dummy inference (B=1). Returns true on success and fills shapes.
    bool runDummy(std::vector<int64_t>& vibShape, std::vector<int64_t>& audShape);

    // Convenience: run a self-test, print basic info and shapes.
    static void SelfTest();

private:
    struct Impl;
    Impl* impl_;
};

