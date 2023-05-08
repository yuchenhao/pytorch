//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mps/MPSGraphVenturaOps.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_copy_from_and_resize.h>
#include <ATen/ops/acos_native.h>
#include <ATen/ops/acosh_native.h>
#include <ATen/ops/asin_native.h>
#include <ATen/ops/asinh_native.h>
#include <ATen/ops/atan_native.h>
#include <ATen/ops/atanh_native.h>
#include <ATen/ops/ceil_native.h>
#include <ATen/ops/cos_native.h>
#include <ATen/ops/cosh_native.h>
#include <ATen/ops/cumsum_native.h>
#include <ATen/ops/erf_native.h>
#include <ATen/ops/erfinv_native.h>
#include <ATen/ops/exp2_native.h>
#include <ATen/ops/exp_native.h>
#include <ATen/ops/expm1_native.h>
#include <ATen/ops/floor_native.h>
#include <ATen/ops/frac_native.h>
#include <ATen/ops/log10_native.h>
#include <ATen/ops/log1p_native.h>
#include <ATen/ops/log2_native.h>
#include <ATen/ops/log_native.h>
#include <ATen/ops/logit_backward_native.h>
#include <ATen/ops/neg_native.h>
#include <ATen/ops/reciprocal_native.h>
#include <ATen/ops/round_native.h>
#include <ATen/ops/rsqrt_native.h>
#include <ATen/ops/sigmoid_native.h>
#include <ATen/ops/sign_native.h>
#include <ATen/ops/signbit_native.h>
#include <ATen/ops/sin_native.h>
#include <ATen/ops/sinh_native.h>
#include <ATen/ops/sqrt_native.h>
#include <ATen/ops/tan_native.h>
#include <ATen/ops/tanh_native.h>
#include <ATen/ops/trunc_native.h>
#endif

namespace at::native {
namespace mps {

typedef MPSGraphTensor* (^UnaryOpBlock)(MPSGraph*, MPSGraphTensor*);
using is_noop_p = std::function<bool(const Tensor&)>;

bool is_empty_tensor(const Tensor& self) {
  return self.numel() == 0;
}

void unary_op(const Tensor& self,
              const Tensor& output,
              std::string op_name,
              UnaryOpBlock unaryBlock,
              is_noop_p is_noop = is_empty_tensor) {
  TORCH_CHECK(!(!is_macos_13_or_newer() && self.scalar_type() == ScalarType::Byte),
              "MPS support unary op with uint8 natively starting from macOS 13.0");
  if (!output.is_same_size(self)) {
    output.resize_(self.sizes());
  }
  if (is_noop(self)) {
    output.copy_(self);
    return;
  }
  @autoreleasepool {
    string key = op_name + getTensorsStringKey({self, output});
    auto cachedGraph = LookUpOrCreateCachedGraph<MPSUnaryCachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      newCachedGraph->inputTensor_ = mpsGraphRankedPlaceHolder(mpsGraph, self);
      MPSGraphTensor* castTensor = newCachedGraph->inputTensor_;
      // Integer input must be cast to float if output is float
      if (isIntegralType(self.scalar_type(), true) && isFloatingType(output.scalar_type())) {
        castTensor = castMPSTensor(mpsGraph, newCachedGraph->inputTensor_, output.scalar_type());
      }
      newCachedGraph->outputTensor_ = unaryBlock(mpsGraph, castTensor);
    });

    bool gatherTensorData = true;
    if (!output.is_contiguous() || output.is_view()) {
      gatherTensorData = false;
    }

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self, /*mpsShape=*/nullptr, gatherTensorData);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output, /*mpsShape=*/nullptr, false);
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =
        @{selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData()};
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
        @{outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()};
    runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, results);
  }
}

MPSGraphTensor* trunc_tensor(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
  // Rounding is a no-op for integral types, and also a reasonable workaround
  // For MPSGraph bug on Apple Silicon, that throws `Function floorOp_i64 was not found in the library`
  // See https://github.com/pytorch/pytorch/issues/84995
  bool isFloatInput = ([inputTensor dataType] & MPSDataTypeFloatBit) != 0;
  if (!isFloatInput) {
    return inputTensor;
  }

  if (!is_macos_13_or_newer()) {
    MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0.0 dataType:inputTensor.dataType];
    MPSGraphTensor* predicateTensor = [mpsGraph lessThanWithPrimaryTensor:inputTensor
                                                          secondaryTensor:zeroTensor
                                                                     name:nil];
    return [mpsGraph selectWithPredicateTensor:predicateTensor
                           truePredicateTensor:[mpsGraph ceilWithTensor:inputTensor name:nil]
                          falsePredicateTensor:[mpsGraph floorWithTensor:inputTensor name:nil]
                                          name:nil];
  } else {
    return [mpsGraph truncateWithTensor:inputTensor name:nil];
  }
};

MPSGraphTensor* log1p(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
  MPSGraphTensor* oneTensor = [mpsGraph constantWithScalar:1.0 dataType:inputTensor.dataType];
  MPSGraphTensor* addedTensor = [mpsGraph additionWithPrimaryTensor:inputTensor secondaryTensor:oneTensor name:nil];
  return [mpsGraph logarithmWithTensor:addedTensor name:nil];
}

} // namespace mps

TORCH_IMPL_FUNC(trunc_out_mps)(const Tensor& self, const Tensor& output) {
  mps::unary_op(self, output, "trunc_out_mps", ^MPSGraphTensor*(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
    return mps::trunc_tensor(mpsGraph, inputTensor);
  });
}

TORCH_IMPL_FUNC(signbit_out_mps)(const Tensor& self, const Tensor& output) {
  mps::unary_op(self, output, "signbit_out_mps", ^MPSGraphTensor*(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
    MPSGraphTensor* output;
    // signbit is not implemented for int64 type.
    // workaround for `Function signbitOp_i64 was not found in the library`
    if ([inputTensor dataType] == MPSDataTypeInt64) {
      MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0.0 dataType:inputTensor.dataType];
      output = [mpsGraph lessThanWithPrimaryTensor:inputTensor secondaryTensor:zeroTensor name:nil];
    } else {
      output = [mpsGraph signbitWithTensor:inputTensor name:nil];
    }
    return mps::castMPSTensor(mpsGraph, output, ScalarType::Bool);
  });
}

TORCH_IMPL_FUNC(sign_out_mps)(const Tensor& self, const Tensor& output) {
  mps::unary_op(self, output, "sign_out_mps", ^MPSGraphTensor*(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
    // Sign op is not implemented in MPS as of MacOS13.0 beta, so simulate it using clamp
    if ([inputTensor dataType] == MPSDataTypeInt64) {
      return [mpsGraph clampWithTensor:inputTensor
                        minValueTensor:[mpsGraph constantWithScalar:-1 dataType:MPSDataTypeInt64]
                        maxValueTensor:[mpsGraph constantWithScalar:1 dataType:MPSDataTypeInt64]
                                  name:nil];
    }
    return [mpsGraph signWithTensor:inputTensor name:nil];
  });
}

#define CREATE_MPS_STRUCTURED_UNARY_ROUNDING_TORCH_IMPL_FUNC(func_out, func_stub)                         \
  TORCH_IMPL_FUNC(func_out)(const Tensor& self, const Tensor& output) {                                   \
    mps::unary_op(                                                                                        \
        self,                                                                                             \
        output,                                                                                           \
        #func_out,                                                                                        \
        ^MPSGraphTensor*(MPSGraph * mpsGraph, MPSGraphTensor * inputTensor) {                             \
          return [mpsGraph func_stub##WithTensor:inputTensor name:nil];                                   \
        },                                                                                                \
        [](const Tensor& t) -> bool { return t.numel() == 0 || isIntegralType(t.scalar_type(), true); }); \
  }
CREATE_MPS_STRUCTURED_UNARY_ROUNDING_TORCH_IMPL_FUNC(ceil_out_mps, ceil)
CREATE_MPS_STRUCTURED_UNARY_ROUNDING_TORCH_IMPL_FUNC(floor_out_mps, floor)
CREATE_MPS_STRUCTURED_UNARY_ROUNDING_TORCH_IMPL_FUNC(round_out_mps, round)

#define CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(func_out, func_stub)                                         \
  TORCH_IMPL_FUNC(func_out)(const Tensor& self, const Tensor& output) {                                          \
    mps::unary_op(self, output, #func_out, ^MPSGraphTensor*(MPSGraph * mpsGraph, MPSGraphTensor * inputTensor) { \
      return [mpsGraph func_stub##WithTensor:inputTensor name:nil];                                              \
    });                                                                                                          \
  }

#define CREATE_MPS_UNARY_TORCH_IMPL_FUNC(func_out, func_stub)                                                    \
  Tensor& func_out(const Tensor& self, Tensor& output) {                                                         \
    mps::unary_op(self, output, #func_out, ^MPSGraphTensor*(MPSGraph * mpsGraph, MPSGraphTensor * inputTensor) { \
      return [mpsGraph func_stub##WithTensor:inputTensor name:nil];                                              \
    });                                                                                                          \
    return output;                                                                                               \
  }

CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(exp_out_mps, exponent)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(exp2_out_mps, exponentBase2)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(reciprocal_out_mps, reciprocal)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(sqrt_out_mps, squareRoot)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(rsqrt_out_mps, reverseSquareRoot)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(neg_out_mps, negative)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(log_out_mps, logarithm)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(log10_out_mps, logarithmBase10)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(log2_out_mps, logarithmBase2)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(erf_out_mps, erf)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(sin_out_mps, sin)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(cos_out_mps, cos)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(tan_out_mps, tan)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(asin_out_mps, asin)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(acos_out_mps, acos)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(atan_out_mps, atan)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(sinh_out_mps, sinh)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(cosh_out_mps, cosh)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(tanh_out_mps, tanh)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(asinh_out_mps, asinh)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(acosh_out_mps, acosh)
CREATE_MPS_STRUCTURED_UNARY_TORCH_IMPL_FUNC(atanh_out_mps, atanh)

CREATE_MPS_UNARY_TORCH_IMPL_FUNC(abs_out_mps, absolute)

Tensor& logical_not_out_mps(const Tensor& self, Tensor& output) {
  auto bool_self = self.to(ScalarType::Bool);
  mps::unary_op(bool_self, output, "logical_not_out_mps", [](MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
    return [mpsGraph notWithTensor:inputTensor name:nil];
  });
  return output;
}

TORCH_IMPL_FUNC(sigmoid_out_mps)(const Tensor& self, const Tensor& output) {
  TORCH_CHECK(self.scalar_type() != ScalarType::Long, "MPS does not support sigmoid op with int64 input");
  mps::unary_op(self, output, "sigmoid_out_mps", ^MPSGraphTensor*(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
    return [mpsGraph sigmoidWithTensor:inputTensor name:nil];
  });
}

TORCH_IMPL_FUNC(log1p_out_mps)(const Tensor& self, const Tensor& output) {
  TORCH_CHECK(self.scalar_type() != ScalarType::Long, "MPS does not support log1p op with int64 input");
  mps::unary_op(self, output, "log1p_out_mps", ^MPSGraphTensor*(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
    return mps::log1p(mpsGraph, inputTensor);
  });
}

TORCH_IMPL_FUNC(frac_out_mps)(const Tensor& self, const Tensor& output) {
  TORCH_CHECK(isFloatingType(self.scalar_type()), "frac_out_mps is only implemented for floating types");
  mps::unary_op(self, output, "frac_out_mps", ^MPSGraphTensor*(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
    auto zeroTensor = [mpsGraph constantWithScalar:0.0 dataType:inputTensor.dataType];
    auto predicateTensor = [mpsGraph lessThanWithPrimaryTensor:inputTensor secondaryTensor:zeroTensor name:nil];
    auto truncTensor = [mpsGraph selectWithPredicateTensor:predicateTensor
                                       truePredicateTensor:[mpsGraph ceilWithTensor:inputTensor name:nil]
                                      falsePredicateTensor:[mpsGraph floorWithTensor:inputTensor name:nil]
                                                      name:nil];
    return [mpsGraph subtractionWithPrimaryTensor:inputTensor secondaryTensor:truncTensor name:nil];
  });
}

TORCH_IMPL_FUNC(erfinv_out_mps)(const Tensor& self, const Tensor& output) {
  // TORCH_CHECK(isFloatingType(self.scalar_type()), "erfinv_out_mps is only implemented for floating types");
  mps::unary_op(self, output, "erfinv_out_mps", ^MPSGraphTensor*(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
    auto negOneTensor = [mpsGraph constantWithScalar:-1.0 dataType:inputTensor.dataType];
    auto zeroTensor = [mpsGraph constantWithScalar:0.0 dataType:inputTensor.dataType];
    auto halfTensor = [mpsGraph constantWithScalar:0.5 dataType:inputTensor.dataType];
    auto oneTensor = [mpsGraph constantWithScalar:1.0 dataType:inputTensor.dataType];
    auto twoTensor = [mpsGraph constantWithScalar:2.0 dataType:inputTensor.dataType];
    auto piTensor = [mpsGraph constantWithScalar:3.14159265358979323846264338327950288 dataType:inputTensor.dataType];
    auto aTensor = [mpsGraph constantWithScalar:0.147 dataType:inputTensor.dataType];
    auto piSquareRootTensor = [mpsGraph constantWithScalar:1.77245385090551602729816748334114518
                                                  dataType:inputTensor.dataType];
    auto infinityTensor = [mpsGraph constantWithScalar:INFINITY dataType:inputTensor.dataType];
    auto negInfinityTensor = [mpsGraph constantWithScalar:-1.0 * INFINITY dataType:inputTensor.dataType];
    auto A = [mpsGraph multiplicationWithPrimaryTensor:inputTensor secondaryTensor:inputTensor name:nil];
    auto B = [mpsGraph logarithmWithTensor:[mpsGraph subtractionWithPrimaryTensor:oneTensor secondaryTensor:A name:nil]
                                      name:nil];
    auto C = [mpsGraph
        additionWithPrimaryTensor:[mpsGraph divisionWithPrimaryTensor:twoTensor
                                                      secondaryTensor:[mpsGraph multiplicationWithPrimaryTensor:piTensor
                                                                                                secondaryTensor:aTensor
                                                                                                           name:nil]
                                                                 name:nil]
                  secondaryTensor:[mpsGraph multiplicationWithPrimaryTensor:B secondaryTensor:halfTensor name:nil]
                             name:nil];
    auto CSquared = [mpsGraph multiplicationWithPrimaryTensor:C secondaryTensor:C name:nil];
    auto CSquaredMinusBDivA = [mpsGraph subtractionWithPrimaryTensor:CSquared
                                                     secondaryTensor:[mpsGraph divisionWithPrimaryTensor:B
                                                                                         secondaryTensor:aTensor
                                                                                                    name:nil]
                                                                name:nil];
    auto squareRootDiffTerm = [mpsGraph squareRootWithTensor:CSquaredMinusBDivA name:nil];
    auto finalDiff = [mpsGraph subtractionWithPrimaryTensor:squareRootDiffTerm secondaryTensor:C name:nil];
    auto finalSquareRoot = [mpsGraph squareRootWithTensor:finalDiff name:nil];
    auto predicateTensor = [mpsGraph greaterThanOrEqualToWithPrimaryTensor:inputTensor
                                                           secondaryTensor:zeroTensor
                                                                      name:nil];
    auto resultPositive = [mpsGraph multiplicationWithPrimaryTensor:finalSquareRoot secondaryTensor:oneTensor name:nil];
    auto resultNegative = [mpsGraph multiplicationWithPrimaryTensor:finalSquareRoot
                                                    secondaryTensor:negOneTensor
                                                               name:nil];
    auto estimated = [mpsGraph selectWithPredicateTensor:predicateTensor
                                     truePredicateTensor:resultPositive
                                    falsePredicateTensor:resultNegative
                                                    name:nil];
    // add 2 steps of Newton-Raphson iteration to improve accuracy
    // adopted from   x = x - (std::erf(x) - y) /
    // ((static_cast<T>(2.0)/static_cast<T>(std::sqrt(c10::pi<double>)))*std::exp(-x*x)); pass 1
    auto negEstimated = [mpsGraph multiplicationWithPrimaryTensor:estimated secondaryTensor:negOneTensor name:nil];
    auto estimatedSquared = [mpsGraph multiplicationWithPrimaryTensor:negEstimated secondaryTensor:estimated name:nil];
    auto estimatedSquaredExp = [mpsGraph exponentWithTensor:estimatedSquared name:nil];
    auto twoDivSquareRootPi = [mpsGraph divisionWithPrimaryTensor:twoTensor
                                                  secondaryTensor:piSquareRootTensor
                                                             name:nil];
    auto gradientDenominator = [mpsGraph multiplicationWithPrimaryTensor:twoDivSquareRootPi
                                                         secondaryTensor:estimatedSquaredExp
                                                                    name:nil];
    auto changeErf = [mpsGraph subtractionWithPrimaryTensor:[mpsGraph erfWithTensor:estimated name:nil]
                                            secondaryTensor:inputTensor
                                                       name:nil];
    auto gradient = [mpsGraph divisionWithPrimaryTensor:changeErf secondaryTensor:gradientDenominator name:nil];
    // pass 2
    auto newEstimated = [mpsGraph subtractionWithPrimaryTensor:estimated secondaryTensor:gradient name:nil];
    auto negEstimated2 = [mpsGraph multiplicationWithPrimaryTensor:newEstimated secondaryTensor:negOneTensor name:nil];
    auto estimatedSquared2 = [mpsGraph multiplicationWithPrimaryTensor:negEstimated2
                                                       secondaryTensor:newEstimated
                                                                  name:nil];
    auto estimatedSquaredExp2 = [mpsGraph exponentWithTensor:estimatedSquared2 name:nil];
    auto twoDivSquareRootPi2 = [mpsGraph divisionWithPrimaryTensor:twoTensor
                                                   secondaryTensor:piSquareRootTensor
                                                              name:nil];
    auto gradientDenominator2 = [mpsGraph multiplicationWithPrimaryTensor:twoDivSquareRootPi2
                                                          secondaryTensor:estimatedSquaredExp2
                                                                     name:nil];
    auto changeErf2 = [mpsGraph subtractionWithPrimaryTensor:[mpsGraph erfWithTensor:newEstimated name:nil]
                                             secondaryTensor:inputTensor
                                                        name:nil];
    auto gradient2 = [mpsGraph divisionWithPrimaryTensor:changeErf2 secondaryTensor:gradientDenominator2 name:nil];
    auto newEstimated2 = [mpsGraph subtractionWithPrimaryTensor:newEstimated secondaryTensor:gradient2 name:nil];

    // post processing step to check if we have exactly +1/-1 then we should map to infinity/-infinity
    // this is because the this algorithm might push us on the wrong side of the asymptote due to rounding
    auto onePredicate = [mpsGraph equalWithPrimaryTensor:inputTensor secondaryTensor:oneTensor name:nil];
    auto negOnePredicate = [mpsGraph equalWithPrimaryTensor:inputTensor secondaryTensor:negOneTensor name:nil];

    auto resultWithInfinity = [mpsGraph selectWithPredicateTensor:onePredicate
                                              truePredicateTensor:infinityTensor
                                             falsePredicateTensor:newEstimated2
                                                             name:nil];
    return [mpsGraph selectWithPredicateTensor:negOnePredicate
                           truePredicateTensor:negInfinityTensor
                          falsePredicateTensor:resultWithInfinity
                                          name:nil];
  });
}

TORCH_IMPL_FUNC(expm1_out_mps)(const Tensor& self, const Tensor& output) {
  mps::unary_op(self, output, "expm1_out_mps", ^MPSGraphTensor*(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
    MPSGraphTensor* oneTensor = [mpsGraph constantWithScalar:1.0 shape:@[ @1 ] dataType:inputTensor.dataType];
    MPSGraphTensor* ePowTensor = [mpsGraph exponentWithTensor:inputTensor name:nil];
    return [mpsGraph subtractionWithPrimaryTensor:ePowTensor secondaryTensor:oneTensor name:nil];
  });
}

void logit_mps_impl(const Tensor& self, c10::optional<double> eps, Tensor& output, const std::string op_name) {
  std::string key = op_name + ":[" + (eps.has_value() ? std::to_string(eps.value()) : "NULL") + "]";

  mps::unary_op(self, output, key, ^MPSGraphTensor*(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
    MPSGraphTensor* oneTensor = [mpsGraph constantWithScalar:1.0 shape:@[ @1 ] dataType:inputTensor.dataType];
    MPSGraphTensor* logitInputTensor;

    if (eps.has_value()) {
      MPSGraphTensor* lowTensor = [mpsGraph constantWithScalar:eps.value() shape:@[ @1 ] dataType:inputTensor.dataType];
      MPSGraphTensor* highTensor = [mpsGraph subtractionWithPrimaryTensor:oneTensor secondaryTensor:lowTensor name:nil];
      logitInputTensor = [mpsGraph clampWithTensor:inputTensor
                                    minValueTensor:lowTensor
                                    maxValueTensor:highTensor
                                              name:nil];
    } else {
      logitInputTensor = inputTensor;
    }

    MPSGraphTensor* oneMinusInputTensor = [mpsGraph subtractionWithPrimaryTensor:oneTensor
                                                                 secondaryTensor:logitInputTensor
                                                                            name:nil];
    MPSGraphTensor* outputTensor = [mpsGraph divisionWithPrimaryTensor:logitInputTensor
                                                       secondaryTensor:oneMinusInputTensor
                                                                  name:nil];
    return [mpsGraph logarithmWithTensor:outputTensor name:nil];
  });
}

Tensor& logit_out_mps(const Tensor& self, c10::optional<double> eps, Tensor& result) {
  logit_mps_impl(self, eps, result, "logit_out_mps");
  return result;
}

Tensor logit_mps(const Tensor& self, c10::optional<double> eps) {
  Tensor result = at::empty(self.sizes(), ScalarType::Float, c10::nullopt, kMPS, c10::nullopt, c10::nullopt);
  logit_mps_impl(self, eps, result, "logit_mps");
  return result;
}

TORCH_IMPL_FUNC(logit_backward_out_mps)
(const Tensor& grad_output, const Tensor& input, c10::optional<double> eps, const Tensor& grad_input) {
  using namespace mps;
  using CachedGraph = MPSUnaryGradCachedGraph;

  // Empty output
  if (grad_input.numel() == 0)
    return;

  double eps_ = eps ? eps.value() : -1.0;

  MPSStream* stream = getCurrentMPSStream();

  @autoreleasepool {
    std::string key = "logit_backward_out_mps:" + getTensorsStringKey({grad_output, input}) + ":" + "[" +
        (eps.has_value() ? std::to_string(eps.value()) : "-1") + "]";

    auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, [&](auto mpsGraph, auto newCachedGraph) {
      MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, input);
      MPSGraphTensor* gradOutputTensor = mpsGraphRankedPlaceHolder(mpsGraph, grad_output);
      MPSGraphTensor* outputTensor = mpsGraphRankedPlaceHolder(mpsGraph, grad_input);
      MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0.0 shape:@[ @1 ] dataType:inputTensor.dataType];
      MPSGraphTensor* oneTensor = [mpsGraph constantWithScalar:1.0 shape:@[ @1 ] dataType:inputTensor.dataType];
      MPSGraphTensor* lowTensor = [mpsGraph constantWithScalar:eps_ shape:@[ @1 ] dataType:inputTensor.dataType];
      MPSGraphTensor* inputLessThanLowPredicateTensor = [mpsGraph lessThanWithPrimaryTensor:inputTensor
                                                                            secondaryTensor:lowTensor
                                                                                       name:nil];
      MPSGraphTensor* highTensor = [mpsGraph subtractionWithPrimaryTensor:oneTensor secondaryTensor:lowTensor name:nil];
      MPSGraphTensor* inputGreaterThanHighPredicateTensor = [mpsGraph greaterThanWithPrimaryTensor:inputTensor
                                                                                   secondaryTensor:highTensor
                                                                                              name:nil];
      MPSGraphTensor* outOfIntervalTensor = [mpsGraph logicalORWithPrimaryTensor:inputLessThanLowPredicateTensor
                                                                 secondaryTensor:inputGreaterThanHighPredicateTensor
                                                                            name:nil];
      MPSGraphTensor* oneMinusInputTensor = [mpsGraph subtractionWithPrimaryTensor:oneTensor
                                                                   secondaryTensor:inputTensor
                                                                              name:nil];
      outputTensor = [mpsGraph multiplicationWithPrimaryTensor:inputTensor
                                               secondaryTensor:oneMinusInputTensor
                                                          name:nil];
      outputTensor = [mpsGraph divisionWithPrimaryTensor:gradOutputTensor secondaryTensor:outputTensor name:nil];
      outputTensor = [mpsGraph selectWithPredicateTensor:outOfIntervalTensor
                                     truePredicateTensor:zeroTensor
                                    falsePredicateTensor:outputTensor
                                                    name:nil];

      newCachedGraph->gradOutputTensor_ = gradOutputTensor;
      newCachedGraph->inputTensor_ = inputTensor;
      newCachedGraph->gradInputTensor_ = outputTensor;
    });
    Placeholder gradOutputPlaceholder = Placeholder(cachedGraph->gradOutputTensor_, grad_output);
    Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor_, input);
    Placeholder gradInputPlaceholder = Placeholder(cachedGraph->gradInputTensor_, grad_input);

    // Create dictionary of inputs and outputs
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      gradOutputPlaceholder.getMPSGraphTensor() : gradOutputPlaceholder.getMPSGraphTensorData(),
      inputPlaceholder.getMPSGraphTensor() : inputPlaceholder.getMPSGraphTensorData(),
    };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
        @{gradInputPlaceholder.getMPSGraphTensor() : gradInputPlaceholder.getMPSGraphTensorData()};
    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }
}

TORCH_IMPL_FUNC(cumsum_out_mps)
(const Tensor& self, int64_t dim, c10::optional<ScalarType> dtype, const Tensor& result) {
  bool macOS13_3_plus = is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_3_PLUS);
  auto nDims = self.dim();
  auto wrapped_dim = maybe_wrap_dim(dim, nDims);
  TORCH_CHECK(wrapped_dim >= 0 && wrapped_dim < std::max(1LL, self.ndimension()),
              "Expected wrapped dim to be between 0 and ",
              self.ndimension(),
              " but got ",
              wrapped_dim,
              "(original dim is ",
              dim,
              ")");
  if (!is_macos_13_or_newer()) {
    TORCH_WARN_ONCE("torch.cumsum supported by MPS on MacOS 13+, please upgrade");
    auto cpu_result = self.to(at::Device(kCPU)).cumsum(dim, dtype);
    at::_copy_from_and_resize(cpu_result, result);
    return;
  }
  auto input = dtype.has_value() ? self.to(dtype.value()) : self;

  // issue #103810551: cumsum is horribly broken for int8, int16 and as chances for overflow is pretty high, cast to
  // int32 fixed in macOS 13.3
  bool castInputData = (isIntegralType(input.scalar_type(), false) && input.scalar_type() != ScalarType::Int &&
                        input.scalar_type() != ScalarType::Long);

  TORCH_CHECK(macOS13_3_plus || input.scalar_type() != ScalarType::Long,
              "MPS does not support cumsum op with int64 input. Support has been added in macOS 13.3");

  mps::unary_op(input,
                result,
                "cumsum_out_mp" + std::to_string(dim),
                ^MPSGraphTensor*(MPSGraph* mpsGraph, MPSGraphTensor* inputTensor) {
                  if (castInputData) {
                    inputTensor = mps::castMPSTensor(mpsGraph, inputTensor, ScalarType::Int);
                  }
                  auto rc = [mpsGraph cumulativeSumWithTensor:inputTensor axis:dim name:nil];
                  if ((mps::getMPSDataType(result) != [rc dataType]) || castInputData) {
                    return mps::castMPSTensor(mpsGraph, rc, result.scalar_type());
                  }
                  return rc;
                });
}

} // namespace at::native
