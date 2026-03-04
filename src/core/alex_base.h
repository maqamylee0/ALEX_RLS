// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/* This file contains the classes for linear models and model builders, helpers
 * for the bitmap,
 * cost model weights, statistic accumulators for collecting cost model
 * statistics,
 * and other miscellaneous functions
 */

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <bitset>
#include <cassert>
#ifdef _WIN32
#include <intrin.h>
#include <limits.h>
typedef unsigned __int32 uint32_t;
#else
#include <stdint.h>
#endif

#ifdef _MSC_VER
#define forceinline __forceinline
#elif defined(__GNUC__)
#define forceinline inline __attribute__((__always_inline__))
#elif defined(__CLANG__)
#if __has_attribute(__always_inline__)
#define forceinline inline __attribute__((__always_inline__))
#else
#define forceinline inline
#endif
#else
#define forceinline inline
#endif

namespace alex {

/*** Linear model and model builder ***/

// Forward declaration
template <class T>
class LinearModelBuilder;

// Linear regression model
template <class T>
class LinearModel {
 public:
  double a_ = 0;  // slope
  double b_ = 0;  // intercept

  LinearModel() = default;
  LinearModel(double a, double b) : a_(a), b_(b) {}
  explicit LinearModel(const LinearModel& other) : a_(other.a_), b_(other.b_) {}

  void expand(double expansion_factor) {
    a_ *= expansion_factor;
    b_ *= expansion_factor;
  }

  inline int predict(T key) const {
    return static_cast<int>(a_ * static_cast<double>(key) + b_);
  }

  inline double predict_double(T key) const {
    return a_ * static_cast<double>(key) + b_;
  }
};

template <class T>
class LinearModelBuilder {
 public:
  LinearModel<T>* model_;

  // RLS and EWC state
  double P_[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
  double lambda_ = 1.0;
  double ewc_lambda_ = 100.0;
  double anchor_a_ = 0.0;
  double anchor_b_ = 0.0;
  bool is_rls_initialized_ = false;

  explicit LinearModelBuilder<T>(LinearModel<T>* model) : model_(model) {}

  inline void add(T x, int y) {
    count_++;
    x_sum_ += static_cast<long double>(x);
    y_sum_ += static_cast<long double>(y);
    xx_sum_ += static_cast<long double>(x) * x;
    xy_sum_ += static_cast<long double>(x) * y;
    x_min_ = std::min<T>(x, x_min_);
    x_max_ = std::max<T>(x, x_max_);
    y_min_ = std::min<double>(y, y_min_);
    y_max_ = std::max<double>(y, y_max_);
  }

  void build() {
    if (count_ <= 1) {
      model_->a_ = 0;
      model_->b_ = static_cast<double>(y_sum_);
      return;
    }

    if (static_cast<long double>(count_) * xx_sum_ - x_sum_ * x_sum_ == 0) {
      // all values in a bucket have the same key.
      model_->a_ = 0;
      model_->b_ = static_cast<double>(y_sum_) / count_;
      return;
    }

    auto slope = static_cast<double>(
        (static_cast<long double>(count_) * xy_sum_ - x_sum_ * y_sum_) /
        (static_cast<long double>(count_) * xx_sum_ - x_sum_ * x_sum_));
    auto intercept = static_cast<double>(
        (y_sum_ - static_cast<long double>(slope) * x_sum_) / count_);
    model_->a_ = slope;
    model_->b_ = intercept;

    // If floating point precision errors, fit spline
    if (model_->a_ <= 0) {
      if (x_max_ - x_min_ == 0){
        model_->a_ = 0;
        model_->b_ = static_cast<double>(y_sum_) / count_;
      }
      else{
        model_->a_ = (y_max_ - y_min_) / (x_max_ - x_min_);
        model_->b_ = -static_cast<double>(x_min_) * model_->a_;
      }
    }
  }

  void init_rls() {
    // Initialize covariance matrix P
    double initial_variance = 1e6; // Large initial uncertainty
    P_[0][0] = initial_variance;
    P_[0][1] = 0;
    P_[1][0] = 0;
    P_[1][1] = initial_variance;

    // Snapshot EWC anchors
    anchor_a_ = model_->a_;
    anchor_b_ = model_->b_;
    is_rls_initialized_ = true;
  }

  void rls_update(T x, int y) {
    if (!is_rls_initialized_) {
      init_rls();
    }

    double x_double = static_cast<double>(x);
    double y_double = static_cast<double>(y);

    // X = [x_double, 1]
    double X[2] = {x_double, 1.0};

    // Calculate predicted y
    double y_pred = model_->a_ * X[0] + model_->b_ * X[1];

    // Compute denominator: lambda + X^T * P * X
    double X_T_P[2];
    X_T_P[0] = X[0] * P_[0][0] + X[1] * P_[1][0];
    X_T_P[1] = X[0] * P_[0][1] + X[1] * P_[1][1];

    double denominator = lambda_ + (X_T_P[0] * X[0] + X_T_P[1] * X[1]);

    if (denominator == 0) return; // Prevent division by zero

    // Compute gain matrix K = (P * X) / denominator
    double K[2];
    K[0] = (P_[0][0] * X[0] + P_[0][1] * X[1]) / denominator;
    K[1] = (P_[1][0] * X[0] + P_[1][1] * X[1]) / denominator;

    // Update weights: theta = theta + K * (y - y_pred)
    double error = y_double - y_pred;
    double new_a = model_->a_ + K[0] * error;
    double new_b = model_->b_ + K[1] * error;

    // Apply EWC regularizer: theta = theta - ewc_lambda * K * P * (theta - anchor) (simplified soft constraint)
    // To keep it computationally cheap and stable, we just do a soft interpolation
    // If ewc_lambda is high, it pulls weights heavily towards the anchor.
    double ewc_pull_a = (new_a - anchor_a_) * (ewc_lambda_ * 1e-4); 
    double ewc_pull_b = (new_b - anchor_b_) * (ewc_lambda_ * 1e-4);

    new_a -= ewc_pull_a;
    new_b -= ewc_pull_b;

    // Bound the slope drift to prevent exponential search breakage
    double min_a = anchor_a_ * 0.5;
    double max_a = anchor_a_ * 2.0;

    if (new_a < min_a) new_a = min_a;
    if (new_a > max_a) new_a = max_a;

    model_->a_ = new_a;
    model_->b_ = new_b;

    // Update covariance matrix: P = (P - K * X^T * P) / lambda
    double K_X_T_P[2][2];
    K_X_T_P[0][0] = K[0] * X_T_P[0];
    K_X_T_P[0][1] = K[0] * X_T_P[1];
    K_X_T_P[1][0] = K[1] * X_T_P[0];
    K_X_T_P[1][1] = K[1] * X_T_P[1];

    P_[0][0] = (P_[0][0] - K_X_T_P[0][0]) / lambda_;
    P_[0][1] = (P_[0][1] - K_X_T_P[0][1]) / lambda_;
    P_[1][0] = (P_[1][0] - K_X_T_P[1][0]) / lambda_;
    P_[1][1] = (P_[1][1] - K_X_T_P[1][1]) / lambda_;
  }

 private:
  int count_ = 0;
  long double x_sum_ = 0;
  long double y_sum_ = 0;
  long double xx_sum_ = 0;
  long double xy_sum_ = 0;
  T x_min_ = std::numeric_limits<T>::max();
  T x_max_ = std::numeric_limits<T>::lowest();
  double y_min_ = std::numeric_limits<double>::max();
  double y_max_ = std::numeric_limits<double>::lowest();
};

/*** Comparison ***/

struct AlexCompare {
  template <class T1, class T2>
  bool operator()(const T1& x, const T2& y) const {
    static_assert(
        std::is_arithmetic<T1>::value && std::is_arithmetic<T2>::value,
        "Comparison types must be numeric.");
    return x < y;
  }
};

/*** Helper methods for bitmap ***/

// Extract the rightmost 1 in the binary representation.
// e.g. extract_rightmost_one(010100100) = 000000100
inline uint64_t extract_rightmost_one(uint64_t value) {
  return value & -static_cast<int64_t>(value);
}

// Remove the rightmost 1 in the binary representation.
// e.g. remove_rightmost_one(010100100) = 010100000
inline uint64_t remove_rightmost_one(uint64_t value) {
  return value & (value - 1);
}

// Count the number of 1s in the binary representation.
// e.g. count_ones(010100100) = 3
inline int count_ones(uint64_t value) {
  return static_cast<int>(_mm_popcnt_u64(value));
}

// Get the offset of a bit in a bitmap.
// word_id is the word id of the bit in a bitmap
// bit is the word that contains the bit
inline int get_offset(int word_id, uint64_t bit) {
  return (word_id << 6) + count_ones(bit - 1);
}

/*** Cost model weights ***/

// Intra-node cost weights
constexpr double kExpSearchIterationsWeight = 20;
constexpr double kShiftsWeight = 0.5;

// TraverseToLeaf cost weights
constexpr double kNodeLookupsWeight = 20;
constexpr double kModelSizeWeight = 5e-7;

/*** Stat Accumulators ***/

struct DataNodeStats {
  double num_search_iterations = 0;
  double num_shifts = 0;
};

// Used when stats are computed using a sample
struct SampleDataNodeStats {
  double log2_sample_size = 0;
  double num_search_iterations = 0;
  double log2_num_shifts = 0;
};

// Accumulates stats that are used in the cost model, based on the actual vs
// predicted position of a key
class StatAccumulator {
 public:
  virtual ~StatAccumulator() = default;
  virtual void accumulate(int actual_position, int predicted_position) = 0;
  virtual double get_stat() = 0;
  virtual void reset() = 0;
};

// Mean log error represents the expected number of exponential search
// iterations when doing a lookup
class ExpectedSearchIterationsAccumulator : public StatAccumulator {
 public:
  void accumulate(int actual_position, int predicted_position) override {
    cumulative_log_error_ +=
        std::log2(std::abs(predicted_position - actual_position) + 1);
    count_++;
  }

  double get_stat() override {
    if (count_ == 0) return 0;
    return cumulative_log_error_ / count_;
  }

  void reset() override {
    cumulative_log_error_ = 0;
    count_ = 0;
  }

 public:
  double cumulative_log_error_ = 0;
  int count_ = 0;
};

// Mean shifts represents the expected number of shifts when doing an insert
class ExpectedShiftsAccumulator : public StatAccumulator {
 public:
  explicit ExpectedShiftsAccumulator(int data_capacity)
      : data_capacity_(data_capacity) {}

  // A dense region of n keys will contribute a total number of expected shifts
  // of approximately
  // ((n-1)/2)((n-1)/2 + 1) = n^2/4 - 1/4
  // This is exact for odd n and off by 0.25 for even n.
  // Therefore, we track n^2/4.
  void accumulate(int actual_position, int) override {
    if (actual_position > last_position_ + 1) {
      long long dense_region_length = last_position_ - dense_region_start_idx_ + 1;
      num_expected_shifts_ += (dense_region_length * dense_region_length) / 4;
      dense_region_start_idx_ = actual_position;
    }
    last_position_ = actual_position;
    count_++;
  }

  double get_stat() override {
    if (count_ == 0) return 0;
    // first need to accumulate statistics for current packed region
    long long dense_region_length = last_position_ - dense_region_start_idx_ + 1;
    long long cur_num_expected_shifts =
        num_expected_shifts_ + (dense_region_length * dense_region_length) / 4;
    return cur_num_expected_shifts / static_cast<double>(count_);
  }

  void reset() override {
    last_position_ = -1;
    dense_region_start_idx_ = 0;
    num_expected_shifts_ = 0;
    count_ = 0;
  }

 public:
  int last_position_ = -1;
  int dense_region_start_idx_ = 0;
  long long num_expected_shifts_ = 0;
  int count_ = 0;
  int data_capacity_ = -1;  // capacity of node
};

// Combines ExpectedSearchIterationsAccumulator and ExpectedShiftsAccumulator
class ExpectedIterationsAndShiftsAccumulator : public StatAccumulator {
 public:
  ExpectedIterationsAndShiftsAccumulator() = default;
  explicit ExpectedIterationsAndShiftsAccumulator(int data_capacity)
      : data_capacity_(data_capacity) {}

  void accumulate(int actual_position, int predicted_position) override {
    cumulative_log_error_ +=
        std::log2(std::abs(predicted_position - actual_position) + 1);

    if (actual_position > last_position_ + 1) {
      long long dense_region_length = last_position_ - dense_region_start_idx_ + 1;
      num_expected_shifts_ += (dense_region_length * dense_region_length) / 4;
      dense_region_start_idx_ = actual_position;
    }
    last_position_ = actual_position;

    count_++;
  }

  double get_stat() override {
    assert(false);  // this should not be used
    return 0;
  }

  double get_expected_num_search_iterations() {
    if (count_ == 0) return 0;
    return cumulative_log_error_ / count_;
  }

  double get_expected_num_shifts() {
    if (count_ == 0) return 0;
    long long dense_region_length = last_position_ - dense_region_start_idx_ + 1;
    long long cur_num_expected_shifts =
        num_expected_shifts_ + (dense_region_length * dense_region_length) / 4;
    return cur_num_expected_shifts / static_cast<double>(count_);
  }

  void reset() override {
    cumulative_log_error_ = 0;
    last_position_ = -1;
    dense_region_start_idx_ = 0;
    num_expected_shifts_ = 0;
    count_ = 0;
  }

 public:
  double cumulative_log_error_ = 0;
  int last_position_ = -1;
  int dense_region_start_idx_ = 0;
  long long num_expected_shifts_ = 0;
  int count_ = 0;
  int data_capacity_ = -1;  // capacity of node
};

/*** Miscellaneous helpers ***/

// https://stackoverflow.com/questions/364985/algorithm-for-finding-the-smallest-power-of-two-thats-greater-or-equal-to-a-giv
inline int pow_2_round_up(int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x + 1;
}

// https://stackoverflow.com/questions/994593/how-to-do-an-integer-log2-in-c
inline int log_2_round_down(int x) {
  int res = 0;
  while (x >>= 1) ++res;
  return res;
}

// https://stackoverflow.com/questions/1666093/cpuid-implementations-in-c
class CPUID {
  uint32_t regs[4];

 public:
  explicit CPUID(unsigned i, unsigned j) {
#ifdef _WIN32
    __cpuidex((int*)regs, (int)i, (int)j);
#else
    asm volatile("cpuid"
                 : "=a"(regs[0]), "=b"(regs[1]), "=c"(regs[2]), "=d"(regs[3])
                 : "a"(i), "c"(j));
#endif
  }

  const uint32_t& EAX() const { return regs[0]; }
  const uint32_t& EBX() const { return regs[1]; }
  const uint32_t& ECX() const { return regs[2]; }
  const uint32_t& EDX() const { return regs[3]; }
};

// https://en.wikipedia.org/wiki/CPUID#EAX=7,_ECX=0:_Extended_Features
inline bool cpu_supports_bmi() {
  return static_cast<bool>(CPUID(7, 0).EBX() & (1 << 3));
}
}