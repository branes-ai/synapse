#pragma once
/// @file gguf.hpp
/// @brief Umbrella header for the GGUF loader.
///
///   auto file = branes::synapse::gguf::mapped_file::open("model.gguf");
///   if (!file) { /* file.error() */ }
///   auto model = branes::synapse::gguf::parse(file->bytes());
///   if (!model) { /* model.error().code, model.error().offset */ }
///   auto arch = model->get<std::string_view>("general.architecture");
///   for (const auto& t : model->tensors()) { auto bytes = model->tensor_data(t); }
///
/// Keep the mapped_file alive while the parsed model is in use: names,
/// strings, and tensor data are views into the mapping.

#include <branes/synapse/gguf/error.hpp>
#include <branes/synapse/gguf/mapped_file.hpp>
#include <branes/synapse/gguf/metadata.hpp>
#include <branes/synapse/gguf/reader.hpp>
#include <branes/synapse/gguf/tensor_type.hpp>
