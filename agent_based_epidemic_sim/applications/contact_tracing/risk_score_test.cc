// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "agent_based_epidemic_sim/applications/contact_tracing/risk_score.h"

#include <vector>

#include "absl/time/time.h"
#include "agent_based_epidemic_sim/applications/contact_tracing/config.pb.h"
#include "agent_based_epidemic_sim/applications/home_work/location_type.h"
#include "agent_based_epidemic_sim/core/parse_text_proto.h"
#include "agent_based_epidemic_sim/core/risk_score.h"
#include "agent_based_epidemic_sim/core/timestep.h"
#include "agent_based_epidemic_sim/port/status_matchers.h"
#include "agent_based_epidemic_sim/util/ostream_overload.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace abesim {
namespace {

using testing::Eq;

absl::Time TimeFromDayAndHour(const int day, const int hour) {
  return absl::UnixEpoch() + absl::Hours(24 * day + hour);
}
absl::Time TimeFromDay(const int day) { return TimeFromDayAndHour(day, 0); }

std::vector<float> FrequencyAdjustments(RiskScore& risk_score,
                                        absl::Span<const Exposure> exposures,
                                        const LocationType type) {
  int64 location_uuid = type == LocationType::kWork ? 0 : 1;

  auto exposure = exposures.begin();
  std::vector<float> adjustments;
  for (const int day : {1, 3, 5, 10, 15, 20, 25}) {
    Timestep timestep(TimeFromDay(day), absl::Hours(24));
    while (exposure != exposures.end() &&
           timestep.end_time() > exposure->start_time) {
      risk_score.AddExposureNotification({.exposure = *exposure},
                                         {.probability = 1.0});
      exposure++;
    }
    adjustments.push_back(risk_score.GetVisitAdjustment(timestep, location_uuid)
                              .frequency_adjustment);
  }
  return adjustments;
}

class RiskScoreTest : public testing::Test {
 protected:
  std::unique_ptr<RiskScore> GetRiskScore() {
    auto risk_score_or = CreateTracingRiskScore(
        GetTracingPolicyProto(), [](const int64 location_uuid) {
          return location_uuid == 0 ? LocationType::kWork : LocationType::kHome;
        });
    return std::move(risk_score_or.value());
  }

 private:
  TracingPolicyProto GetTracingPolicyProto() {
    return ParseTextProtoOrDie<TracingPolicyProto>(R"(
      test_validity_duration { seconds: 604800 }
      contact_retention_duration { seconds: 1209600 }
      quarantine_duration { seconds: 1209600 }
      test_latency { seconds: 86400 }
      positive_threshold: .9
    )");
  }
};

OVERLOAD_VECTOR_OSTREAM_OPS

struct Case {
  HealthState::State initial_health_state;
  std::vector<Exposure> positive_exposures;
  LocationType location_type;
  std::vector<float> expected_adjustments;

  friend std::ostream& operator<<(std::ostream& strm, const Case& c) {
    return strm << "{" << c.initial_health_state << ", " << c.positive_exposures
                << ", " << static_cast<int>(c.location_type) << ", "
                << c.expected_adjustments << "}";
  }
};

TEST_F(RiskScoreTest, GetVisitAdjustment) {
  Case cases[] = {
      {
          .initial_health_state = HealthState::SUSCEPTIBLE,
          .positive_exposures = {{.start_time = TimeFromDayAndHour(2, 4)}},
          .location_type = LocationType::kWork,
          .expected_adjustments = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0},
      },
      {
          .initial_health_state = HealthState::SUSCEPTIBLE,
          .positive_exposures = {},
          .location_type = LocationType::kWork,
          .expected_adjustments = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
      },
      // We always go to home locations.
      {
          .initial_health_state = HealthState::SUSCEPTIBLE,
          .positive_exposures = {{.start_time = TimeFromDay(1)}},
          .location_type = LocationType::kHome,
          .expected_adjustments = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
      },
      {
          .initial_health_state = HealthState::EXPOSED,
          .positive_exposures = {},
          .location_type = LocationType::kHome,
          .expected_adjustments = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
      },
      {
          .initial_health_state = HealthState::EXPOSED,
          .positive_exposures = {{.start_time = TimeFromDay(1)}},
          .location_type = LocationType::kHome,
          .expected_adjustments = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
      },
      {
          .initial_health_state = HealthState::SUSCEPTIBLE,
          .positive_exposures = {},
          .location_type = LocationType::kHome,
          .expected_adjustments = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
      },
  };

  for (const Case& c : cases) {
    auto risk_score = GetRiskScore();
    risk_score->AddHealthStateTransistion({
        .time = absl::InfinitePast(),
        .health_state = c.initial_health_state,
    });
    auto adjustments = FrequencyAdjustments(*risk_score, c.positive_exposures,
                                            c.location_type);
    EXPECT_THAT(adjustments, testing::ElementsAreArray(c.expected_adjustments))
        << c;
  }
}

TEST_F(RiskScoreTest, GetTestResult) {
  auto risk_score = GetRiskScore();

  {
    // Before there is a test we return the null result.
    TestResult result =
        risk_score->GetTestResult(Timestep(TimeFromDay(1), absl::Hours(24)));
    TestResult expected = {.time_requested = absl::InfiniteFuture(),
                           .time_received = absl::InfiniteFuture(),
                           .probability = 0.0};
    EXPECT_EQ(result, expected);
  }

  risk_score->AddExposureNotification(
      {.exposure = {.start_time = TimeFromDay(1)}},
      {
          .time_requested = TimeFromDay(1),
          .time_received = TimeFromDay(2),
          .probability = 0.0,
      });
  {
    // Negative results don't matter.
    TestResult result =
        risk_score->GetTestResult(Timestep(TimeFromDay(2), absl::Hours(24)));
    TestResult expected = {.time_requested = absl::InfiniteFuture(),
                           .time_received = absl::InfiniteFuture(),
                           .probability = 0.0};
    EXPECT_EQ(result, expected);
  }

  risk_score->AddExposureNotification(
      {.exposure = {.start_time = TimeFromDay(2)}},
      {
          .time_requested = TimeFromDay(2),
          .time_received = TimeFromDay(3),
          .probability = 1.0,
      });
  {
    // On positive contact reports we perform a test, but if we're not sick
    // the result is negative.
    TestResult result =
        risk_score->GetTestResult(Timestep(TimeFromDay(4), absl::Hours(24)));
    TestResult expected = {.time_requested = TimeFromDay(3),
                           .time_received = TimeFromDay(4),
                           .probability = 0.0};
    EXPECT_EQ(result, expected);
  }

  risk_score->AddExposureNotification(
      {.exposure = {.start_time = TimeFromDay(8)}},
      {
          .time_requested = TimeFromDay(8),
          .time_received = TimeFromDay(9),
          .probability = 1.0,
      });
  {
    // Another positive contact that is within the test validity period will
    // NOT cause another test.
    TestResult result =
        risk_score->GetTestResult(Timestep(TimeFromDay(10), absl::Hours(24)));
    TestResult expected = {.time_requested = TimeFromDay(3),
                           .time_received = TimeFromDay(4),
                           .probability = 0.0};
    EXPECT_EQ(result, expected);
  }

  risk_score->AddHealthStateTransistion(
      {.time = TimeFromDay(12), .health_state = HealthState::EXPOSED});
  risk_score->AddExposureNotification(
      {.exposure = {.start_time = TimeFromDay(12)}},
      {
          .time_requested = TimeFromDay(12),
          .time_received = TimeFromDay(13),
          .probability = 1.0,
      });
  {
    // Another positive contact after the validity period expires will perform
    // another test.  This time it will report that we are sick since we
    // have transitioned health states.
    TestResult result =
        risk_score->GetTestResult(Timestep(TimeFromDay(14), absl::Hours(24)));
    TestResult expected = {.time_requested = TimeFromDay(13),
                           .time_received = TimeFromDay(14),
                           .probability = 1.0};
    EXPECT_EQ(result, expected);
  }
}

TEST_F(RiskScoreTest, GetsContactTracingPolicy) {
  auto risk_score = GetRiskScore();

  // If there's no positive test, we don't send.
  EXPECT_THAT(risk_score->GetContactTracingPolicy(
                  Timestep(TimeFromDay(5), absl::Hours(24))),
              Eq(RiskScore::ContactTracingPolicy{.report_recursively = false,
                                                 .send_report = false}));

  risk_score->AddHealthStateTransistion(
      {.time = TimeFromDay(2), .health_state = HealthState::EXPOSED});
  risk_score->AddExposureNotification({}, {
                                              .time_requested = TimeFromDay(3),
                                              .time_received = TimeFromDay(6),
                                              .probability = 1.0,
                                          });

  // If the test isn't received yet (will be recieved on day 7) don't send.
  EXPECT_THAT(risk_score->GetContactTracingPolicy(
                  Timestep(TimeFromDay(5), absl::Hours(24))),
              Eq(RiskScore::ContactTracingPolicy{.report_recursively = false,
                                                 .send_report = false}));
  // The test has been received.
  EXPECT_THAT(risk_score->GetContactTracingPolicy(
                  Timestep(TimeFromDay(7), absl::Hours(24))),
              Eq(RiskScore::ContactTracingPolicy{.report_recursively = false,
                                                 .send_report = true}));
  // Don't send old tests that were requested 2 weeks ago+;
  EXPECT_THAT(risk_score->GetContactTracingPolicy(
                  Timestep(TimeFromDay(21), absl::Hours(24))),
              Eq(RiskScore::ContactTracingPolicy{.report_recursively = false,
                                                 .send_report = false}));
}

TEST_F(RiskScoreTest, GetsContactRetentionDuration) {
  auto risk_score = GetRiskScore();
  EXPECT_EQ(risk_score->ContactRetentionDuration(), absl::Hours(24 * 14));
}

}  // namespace
}  // namespace abesim
