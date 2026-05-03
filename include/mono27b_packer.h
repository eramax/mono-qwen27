#pragma once

#include <string>

bool mono27b_pack_models(const std::string & target_gguf,
                         const std::string & draft_gguf,
                         const std::string & out_blob,
                         std::string & status,
                         std::string & error);
