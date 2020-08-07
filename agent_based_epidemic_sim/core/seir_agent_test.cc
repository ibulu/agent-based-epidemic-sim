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

#include "agent_based_epidemic_sim/core/seir_agent.h"

#include "absl/time/time.h"
#include "agent_based_epidemic_sim/core/broker.h"
#include "agent_based_epidemic_sim/core/constants.h"
#include "agent_based_epidemic_sim/core/event.h"
#include "agent_based_epidemic_sim/core/integral_types.h"
#include "agent_based_epidemic_sim/core/risk_score.h"
#include "agent_based_epidemic_sim/core/timestep.h"
#include "agent_based_epidemic_sim/core/transition_model.h"
#include "agent_based_epidemic_sim/core/visit.h"
#include "agent_based_epidemic_sim/core/visit_generator.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace abesim {
namespace {

using testing::_;
using testing::Eq;
using testing::NotNull;
using testing::Ref;
using testing::Return;
using testing::SetArgPointee;

class MockTransitionModel : public TransitionModel {
 public:
  explicit MockTransitionModel() = default;
  MOCK_METHOD(HealthTransition, GetNextHealthTransition,
              (const HealthTransition& latest_transition), (override));
};

class MockTransmissionModel : public TransmissionModel {
 public:
  MockTransmissionModel() = default;
  MOCK_METHOD(HealthTransition, GetInfectionOutcome,
              (absl::Span<const Exposure* const> exposures), (override));
};

class MockVisitGenerator : public VisitGenerator {
 public:
  explicit MockVisitGenerator() = default;
  MOCK_METHOD(void, GenerateVisits,
              (const Timestep& timestep, const RiskScore& policy,
               std::vector<Visit>* visits),
              (override));
};

template <typename T>
class MockBroker : public Broker<T> {
 public:
  explicit MockBroker() = default;
  MOCK_METHOD(void, Send, (absl::Span<const T> visits), (override));
};

class MockRiskScore : public RiskScore {
 public:
  MOCK_METHOD(void, AddHealthStateTransistion, (HealthTransition transition),
              (override));
  MOCK_METHOD(void, AddExposures, (absl::Span<const Exposure* const> exposures),
              (override));
  MOCK_METHOD(void, AddExposureNotification,
              (const Contact& contact, const TestResult& result), (override));
  MOCK_METHOD(VisitAdjustment, GetVisitAdjustment,
              (const Timestep& timestep, int64 location_uuid),
              (const, override));
  MOCK_METHOD(TestResult, GetTestResult, (const Timestep& timestep),
              (const, override));
  MOCK_METHOD(ContactTracingPolicy, GetContactTracingPolicy,
              (const Timestep& timestep), (const, override));
  MOCK_METHOD(absl::Duration, ContactRetentionDuration, (), (const, override));
};

std::vector<InfectionOutcome> OutcomesFromContacts(
    const int64 agent_uuid, absl::Span<const Contact> contacts) {
  std::vector<InfectionOutcome> outcomes;
  for (const Contact& contact : contacts) {
    outcomes.push_back({
        .agent_uuid = agent_uuid,
        .exposure = contact.exposure,
        .exposure_type = InfectionOutcomeProto::CONTACT,
        .source_uuid = contact.other_uuid,
    });
  }
  return outcomes;
}

TEST(SEIRAgentTest, ComputesVisits) {
  auto transition_model = absl::make_unique<MockTransitionModel>();
  auto visit_generator = absl::make_unique<MockVisitGenerator>();
  auto visit_broker = absl::make_unique<MockBroker<Visit>>();
  MockTransmissionModel transmission_model;
  auto risk_score = NewNullRiskScore();
  const Timestep timestep(absl::UnixEpoch(), absl::Hours(24));
  EXPECT_CALL(*transition_model, GetNextHealthTransition(Eq(HealthTransition{
                                     .time = absl::FromUnixSeconds(-43200LL),
                                     .health_state = HealthState::EXPOSED})))
      .WillOnce(
          Return(HealthTransition{.time = absl::FromUnixSeconds(43200LL),
                                  .health_state = HealthState::INFECTIOUS}));
  EXPECT_CALL(*transition_model, GetNextHealthTransition(Eq(HealthTransition{
                                     .time = absl::FromUnixSeconds(43200LL),
                                     .health_state = HealthState::INFECTIOUS})))
      .Times(1);
  const int64 kUuid = 42LL;
  std::vector<Visit> visits{Visit{.location_uuid = 0LL,
                                  .start_time = absl::FromUnixSeconds(0LL),
                                  .end_time = absl::FromUnixSeconds(28800LL)},
                            Visit{.location_uuid = 1LL,
                                  .start_time = absl::FromUnixSeconds(28800LL),
                                  .end_time = absl::FromUnixSeconds(57600LL)},
                            Visit{.location_uuid = 0LL,
                                  .start_time = absl::FromUnixSeconds(57600LL),
                                  .end_time = absl::FromUnixSeconds(86400LL)}};
  EXPECT_CALL(*visit_generator,
              GenerateVisits(timestep, Ref(*risk_score), NotNull()))
      .WillOnce(SetArgPointee<2>(visits));
  std::vector<Visit> expected_visits{
      Visit{.location_uuid = 0LL,
            .agent_uuid = kUuid,
            .start_time = absl::FromUnixSeconds(0LL),
            .end_time = absl::FromUnixSeconds(28800LL),
            .health_state = HealthState::EXPOSED,
            .infectivity = kInfectivityArray[1]},
      Visit{.location_uuid = 1LL,
            .agent_uuid = kUuid,
            .start_time = absl::FromUnixSeconds(28800LL),
            .end_time = absl::FromUnixSeconds(43200LL),
            .health_state = HealthState::EXPOSED,
            .infectivity = kInfectivityArray[1]},
      Visit{.location_uuid = 0LL,
            .agent_uuid = kUuid,
            .start_time = absl::FromUnixSeconds(57600LL),
            .end_time = absl::FromUnixSeconds(86400LL),
            .health_state = HealthState::INFECTIOUS,
            .infectivity = kInfectivityArray[1]},
      Visit{.location_uuid = 1LL,
            .agent_uuid = kUuid,
            .start_time = absl::FromUnixSeconds(43200LL),
            .end_time = absl::FromUnixSeconds(57600LL),
            .health_state = HealthState::INFECTIOUS,
            .infectivity = kInfectivityArray[1]}};
  EXPECT_CALL(*visit_broker, Send(Eq(expected_visits)));
  auto agent =
      SEIRAgent::Create(kUuid,
                        {.time = absl::FromUnixSeconds(-43200LL),
                         .health_state = HealthState::EXPOSED},
                        &transmission_model, std::move(transition_model),
                        std::move(visit_generator), std::move(risk_score));
  agent->ProcessInfectionOutcomes(timestep, {});
  agent->ComputeVisits(timestep, visit_broker.get());
}

TEST(SEIRAgentTest, InitializesSusceptibleState) {
  auto transition_model = absl::make_unique<MockTransitionModel>();
  auto visit_generator = absl::make_unique<MockVisitGenerator>();
  auto visit_broker = absl::make_unique<MockBroker<Visit>>();
  MockTransmissionModel transmission_model;
  auto risk_score = NewNullRiskScore();
  const Timestep timestep(absl::UnixEpoch(), absl::Hours(24));
  EXPECT_CALL(*transition_model, GetNextHealthTransition).Times(0);
  const int64 kUuid = 42LL;
  std::vector<Visit> visits{Visit{.location_uuid = 0LL,
                                  .start_time = absl::FromUnixSeconds(0LL),
                                  .end_time = absl::FromUnixSeconds(86400LL)}};
  EXPECT_CALL(*visit_generator,
              GenerateVisits(timestep, Ref(*risk_score), NotNull()))
      .WillOnce(SetArgPointee<2>(visits));
  std::vector<Visit> expected_visits{
      Visit{.location_uuid = 0LL,
            .agent_uuid = kUuid,
            .start_time = absl::FromUnixSeconds(0LL),
            .end_time = absl::FromUnixSeconds(86400LL),
            .health_state = HealthState::SUSCEPTIBLE,
            .infectivity = 0.0f}};
  EXPECT_CALL(*visit_broker, Send(Eq(expected_visits)));
  auto agent = SEIRAgent::CreateSusceptible(
      kUuid, &transmission_model, std::move(transition_model),
      std::move(visit_generator), std::move(risk_score));
  agent->ComputeVisits(timestep, visit_broker.get());
}

TEST(SEIRAgentTest, InitializesNonSusceptibleState) {
  auto transition_model = absl::make_unique<MockTransitionModel>();
  auto visit_generator = absl::make_unique<MockVisitGenerator>();
  auto visit_broker = absl::make_unique<MockBroker<Visit>>();
  MockTransmissionModel transmission_model;
  auto risk_score = NewNullRiskScore();
  const Timestep timestep(absl::UnixEpoch(), absl::Hours(24));
  EXPECT_CALL(*transition_model, GetNextHealthTransition(Eq(HealthTransition{
                                     .time = absl::FromUnixSeconds(-1LL),
                                     .health_state = HealthState::EXPOSED})))
      .WillOnce(
          Return(HealthTransition{.time = absl::FromUnixSeconds(86400LL),
                                  .health_state = HealthState::INFECTIOUS}));
  const int64 kUuid = 42LL;
  std::vector<Visit> visits{Visit{.location_uuid = 0LL,
                                  .start_time = absl::FromUnixSeconds(0LL),
                                  .end_time = absl::FromUnixSeconds(86400LL)}};
  EXPECT_CALL(*visit_generator,
              GenerateVisits(timestep, Ref(*risk_score), NotNull()))
      .WillOnce(SetArgPointee<2>(visits));
  std::vector<Visit> expected_visits{
      Visit{.location_uuid = 0LL,
            .agent_uuid = kUuid,
            .start_time = absl::FromUnixSeconds(0LL),
            .end_time = absl::FromUnixSeconds(86400LL),
            .health_state = HealthState::EXPOSED,
            .infectivity = kInfectivityArray[0]}};
  EXPECT_CALL(*visit_broker, Send(Eq(expected_visits)));
  auto agent =
      SEIRAgent::Create(kUuid,
                        {.time = absl::FromUnixSeconds(-1LL),
                         .health_state = HealthState::EXPOSED},
                        &transmission_model, std::move(transition_model),
                        std::move(visit_generator), std::move(risk_score));
  agent->ProcessInfectionOutcomes(timestep, {});
  agent->ComputeVisits(timestep, visit_broker.get());
}

TEST(SEIRAgentTest, RespectsTimestepBasedDwellTimeAndFiltersZeroIntervals) {
  auto transition_model = absl::make_unique<MockTransitionModel>();
  auto visit_generator = absl::make_unique<MockVisitGenerator>();
  auto visit_broker = absl::make_unique<MockBroker<Visit>>();
  MockTransmissionModel transmission_model;
  auto risk_score = NewNullRiskScore();
  const Timestep timestep(absl::UnixEpoch(), absl::Hours(24));
  EXPECT_CALL(*transition_model, GetNextHealthTransition(Eq(HealthTransition{
                                     .time = absl::FromUnixSeconds(-1LL),
                                     .health_state = HealthState::EXPOSED})))
      .WillOnce(
          Return(HealthTransition{.time = absl::FromUnixSeconds(-1LL),
                                  .health_state = HealthState::INFECTIOUS}));
  // Transition is recorded with a forward-adjusted transition time.
  EXPECT_CALL(*transition_model,
              GetNextHealthTransition(Eq(
                  HealthTransition{.time = absl::FromUnixSeconds(86400LL - 1LL),
                                   .health_state = HealthState::INFECTIOUS})))
      .WillOnce(
          Return(HealthTransition{.time = absl::FromUnixSeconds(2LL * 86400LL),
                                  .health_state = HealthState::RECOVERED}));
  const int64 kUuid = 42LL;
  std::vector<Visit> visits{Visit{.location_uuid = 0LL,
                                  .start_time = absl::FromUnixSeconds(0LL),
                                  .end_time = absl::FromUnixSeconds(86400LL)}};
  EXPECT_CALL(*visit_generator,
              GenerateVisits(timestep, Ref(*risk_score), NotNull()))
      .WillOnce(SetArgPointee<2>(visits));
  std::vector<Visit> expected_visits{
      Visit{.location_uuid = 0LL,
            .agent_uuid = kUuid,
            .start_time = absl::FromUnixSeconds(0LL),
            .end_time = absl::FromUnixSeconds(86400LL - 1LL),
            .health_state = HealthState::EXPOSED,
            .infectivity = kInfectivityArray[0]},
      Visit{.location_uuid = 0LL,
            .agent_uuid = kUuid,
            .start_time = absl::FromUnixSeconds(86400LL - 1LL),
            .end_time = absl::FromUnixSeconds(86400LL),
            .health_state = HealthState::INFECTIOUS,
            .infectivity = kInfectivityArray[1]},
  };
  EXPECT_CALL(*visit_broker, Send(Eq(expected_visits)));
  auto agent =
      SEIRAgent::Create(kUuid,
                        {.time = absl::FromUnixSeconds(-1LL),
                         .health_state = HealthState::EXPOSED},
                        &transmission_model, std::move(transition_model),
                        std::move(visit_generator), std::move(risk_score));
  agent->ProcessInfectionOutcomes(timestep, {});
  agent->ComputeVisits(timestep, visit_broker.get());
}

TEST(SEIRAgentTest, ProcessesInfectionOutcomesIgnoresIfAlreadyExposed) {
  auto transition_model = absl::make_unique<MockTransitionModel>();
  EXPECT_CALL(*transition_model, GetNextHealthTransition(Eq(HealthTransition{
                                     .time = absl::FromUnixSeconds(-1LL),
                                     .health_state = HealthState::EXPOSED})))
      .WillOnce(
          Return(HealthTransition{.time = absl::FromUnixSeconds(86400LL),
                                  .health_state = HealthState::INFECTIOUS}));
  auto visit_generator = absl::make_unique<MockVisitGenerator>();
  MockTransmissionModel transmission_model;
  EXPECT_CALL(transmission_model, GetInfectionOutcome(_))
      .Times(1)
      .WillOnce(Return(HealthTransition{.time = absl::FromUnixSeconds(-1LL),
                                        .health_state = HealthState::EXPOSED}));
  auto risk_score = NewNullRiskScore();
  const int64 kUuid = 42LL;

  auto agent = SEIRAgent::CreateSusceptible(
      kUuid, &transmission_model, std::move(transition_model),
      std::move(visit_generator), std::move(risk_score));

  const Timestep timestep(absl::UnixEpoch(), absl::Hours(24));
  {
    std::vector<InfectionOutcome> infection_outcomes{
        InfectionOutcome{.agent_uuid = kUuid,
                         .exposure = {.start_time = absl::FromUnixSeconds(-1LL),
                                      .infectivity = 1.0f},
                         .exposure_type = InfectionOutcomeProto::CONTACT,
                         .source_uuid = 2LL}};
    agent->ProcessInfectionOutcomes(timestep, infection_outcomes);
    EXPECT_EQ(agent->NextHealthTransition().time,
              absl::FromUnixSeconds(86400LL));
  }
  {
    std::vector<InfectionOutcome> infection_outcomes{
        InfectionOutcome{.agent_uuid = kUuid,
                         .exposure = {.start_time = absl::FromUnixSeconds(5LL),
                                      .infectivity = 1.0f},
                         .exposure_type = InfectionOutcomeProto::CONTACT,
                         .source_uuid = 3LL}};
    // A new call with an infection outcome with a different time has no
    // effect, only the first exposure matters.
    agent->ProcessInfectionOutcomes(timestep, infection_outcomes);
    EXPECT_EQ(agent->NextHealthTransition().time,
              absl::FromUnixSeconds(86400LL));
  }
}

TEST(SEIRAgentTest, ProcessesInfectionOutcomesRemainsSusceptible) {
  auto transition_model = absl::make_unique<MockTransitionModel>();
  EXPECT_CALL(*transition_model, GetNextHealthTransition).Times(0);
  auto visit_generator = absl::make_unique<MockVisitGenerator>();
  MockTransmissionModel transmission_model;
  auto risk_score = NewNullRiskScore();
  const int64 kUuid = 42LL;

  EXPECT_CALL(transmission_model, GetInfectionOutcome(_))
      .Times(1)
      .WillOnce(
          Return(HealthTransition{.health_state = HealthState::SUSCEPTIBLE}));
  auto agent = SEIRAgent::CreateSusceptible(
      kUuid, &transmission_model, std::move(transition_model),
      std::move(visit_generator), std::move(risk_score));

  const Timestep timestep(absl::UnixEpoch(), absl::Hours(24));
  std::vector<InfectionOutcome> infection_outcomes{
      InfectionOutcome{.agent_uuid = kUuid,
                       .exposure = {.start_time = absl::FromUnixSeconds(-1LL),
                                    .infectivity = 1.0f},
                       .exposure_type = InfectionOutcomeProto::CONTACT,
                       .source_uuid = 2LL}};
  agent->ProcessInfectionOutcomes(timestep, infection_outcomes);
  EXPECT_THAT(agent->NextHealthTransition(),
              Eq(HealthTransition{.time = absl::InfiniteFuture(),
                                  .health_state = HealthState::SUSCEPTIBLE}));
}

TEST(SEIRAgentTest, ProcessesInfectionOutcomesMultipleExposuresSameContact) {
  auto transition_model = absl::make_unique<MockTransitionModel>();
  EXPECT_CALL(*transition_model, GetNextHealthTransition).Times(0);
  auto visit_generator = absl::make_unique<MockVisitGenerator>();
  MockTransmissionModel transmission_model;
  auto risk_score = NewNullRiskScore();
  const int64 kUuid = 42LL;

  EXPECT_CALL(transmission_model, GetInfectionOutcome(_)).Times(1);
  auto agent = SEIRAgent::CreateSusceptible(
      kUuid, &transmission_model, std::move(transition_model),
      std::move(visit_generator), std::move(risk_score));

  const Timestep timestep(absl::UnixEpoch(), absl::Hours(24));
  std::vector<InfectionOutcome> infection_outcomes{
      InfectionOutcome{.agent_uuid = kUuid,
                       .exposure = {.start_time = absl::FromUnixSeconds(-2LL),
                                    .infectivity = 1.0f},
                       .exposure_type = InfectionOutcomeProto::CONTACT,
                       .source_uuid = 2LL},
      InfectionOutcome{.agent_uuid = kUuid,
                       .exposure = {.start_time = absl::FromUnixSeconds(-1LL),
                                    .infectivity = 1.0f},
                       .exposure_type = InfectionOutcomeProto::CONTACT,
                       .source_uuid = 2LL}};
  agent->ProcessInfectionOutcomes(timestep, infection_outcomes);
}

TEST(SEIRAgentTest, ProcessInfectionOutcomesRejectsWrongUuid) {
  auto transition_model = absl::make_unique<MockTransitionModel>();
  auto visit_generator = absl::make_unique<MockVisitGenerator>();
  MockTransmissionModel transmission_model;
  auto risk_score = NewNullRiskScore();
  const int64 kUuid = 42LL;
  std::vector<InfectionOutcome> infection_outcomes{InfectionOutcome{
      .agent_uuid = kUuid + 1,
  }};
  auto agent = SEIRAgent::CreateSusceptible(
      kUuid, &transmission_model, std::move(transition_model),
      std::move(visit_generator), std::move(risk_score));
  const Timestep timestep(absl::UnixEpoch(), absl::Hours(24));
  ASSERT_DEBUG_DEATH(
      agent->ProcessInfectionOutcomes(timestep, infection_outcomes), "");
}

TEST(SEIRAgentTest, ProcessInfectionOutcomesReturnsNoOpIfNonePresent) {
  auto transition_model = absl::make_unique<MockTransitionModel>();
  auto visit_generator = absl::make_unique<MockVisitGenerator>();
  MockTransmissionModel transmission_model;
  auto risk_score = NewNullRiskScore();
  EXPECT_CALL(*transition_model, GetNextHealthTransition).Times(0);
  const int64 kUuid = 42LL;
  auto agent = SEIRAgent::CreateSusceptible(
      kUuid, &transmission_model, std::move(transition_model),
      std::move(visit_generator), std::move(risk_score));
  const Timestep timestep(absl::UnixEpoch(), absl::Hours(24));
  agent->ProcessInfectionOutcomes(timestep, {});
}

void ExpectAgentInitialization(RiskScore& risk_score) {}

// Test that the agent updates the risk score with newly received exposures and
// contact reports.
TEST(SEIRAgentTest,
     ProcessInfectionOutcomesAndUpdateContactReportsUpdateRiskScore) {
  const int64 kUuid = 42LL;
  auto transition_model = absl::make_unique<MockTransitionModel>();
  auto visit_generator = absl::make_unique<MockVisitGenerator>();
  auto contact_report_broker = absl::make_unique<MockBroker<ContactReport>>();

  std::vector<Contact> contacts = {
      {
          .other_uuid = 12LL,
          .exposure = {.start_time = absl::FromUnixSeconds(0LL),
                       .duration = absl::Hours(1LL),
                       .infectivity = 0.0f},
      },
      {
          .other_uuid = 15LL,
          .exposure = {.start_time = absl::FromUnixSeconds(0LL),
                       .duration = absl::Hours(1LL),
                       .infectivity = 0.0f},
      }};
  std::vector<ContactReport> contact_reports = {
      {.from_agent_uuid = 12LL,
       .to_agent_uuid = kUuid,
       .test_result = {.probability = 0.5}},
      {.from_agent_uuid = 13LL,
       .to_agent_uuid = kUuid,
       .test_result = {.probability = 0.75}},
      {.from_agent_uuid = 15LL,
       .to_agent_uuid = kUuid,
       .test_result = {.probability = 0.95}},
  };
  auto outcomes = OutcomesFromContacts(kUuid, contacts);

  MockTransmissionModel transmission_model;
  EXPECT_CALL(transmission_model,
              GetInfectionOutcome(testing::ElementsAre(&outcomes[0].exposure,
                                                       &outcomes[1].exposure)));

  auto risk_score = absl::make_unique<MockRiskScore>();
  EXPECT_CALL(*risk_score, ContactRetentionDuration())
      .WillRepeatedly(Return(absl::Hours(24 * 14)));
  EXPECT_CALL(*risk_score,
              AddHealthStateTransistion(HealthTransition{
                  absl::InfinitePast(), HealthState::SUSCEPTIBLE}));

  EXPECT_CALL(*risk_score, GetContactTracingPolicy(_))
      .WillOnce(Return(RiskScore::ContactTracingPolicy{}));

  // This is a key assertion that we pass on exposures from
  // ProcessInfectionOutcomes to the risk score.
  EXPECT_CALL(*risk_score, AddExposures(testing::ElementsAre(
                               &outcomes[0].exposure, &outcomes[1].exposure)));

  // This a key assertion that we do pass on contact reports to the
  // risk score.
  // Only the contact_report form agents 12 and 15 are reported because the
  // agent had no matching exposure for agent 13.
  EXPECT_CALL(*risk_score, AddExposureNotification(
                               contacts[0], contact_reports[0].test_result));
  EXPECT_CALL(*risk_score, AddExposureNotification(
                               contacts[1], contact_reports[2].test_result));

  auto agent = SEIRAgent::CreateSusceptible(
      kUuid, &transmission_model, std::move(transition_model),
      std::move(visit_generator), std::move(risk_score));

  Timestep timestep(absl::UnixEpoch(), absl::Hours(24));
  agent->ProcessInfectionOutcomes(timestep, outcomes);
  agent->UpdateContactReports(timestep, contact_reports,
                              contact_report_broker.get());
}

// Test that the agent sends contact reports to the right recipients.
TEST(SEIRAgentTest, UpdateContactReportsSendsReports) {
  const int64 kUuid = 42LL;
  auto transition_model = absl::make_unique<MockTransitionModel>();
  auto visit_generator = absl::make_unique<MockVisitGenerator>();
  auto contact_report_broker = absl::make_unique<MockBroker<ContactReport>>();

  std::vector<Contact> contacts = {
      {
          .other_uuid = 12LL,
          .exposure = {.start_time = absl::FromUnixSeconds(0LL),
                       .duration = absl::Hours(1LL),
                       .infectivity = 0.0f},
      },
      {
          .other_uuid = 15LL,
          .exposure = {.start_time = absl::FromUnixSeconds(0LL),
                       .duration = absl::Hours(1LL),
                       .infectivity = 0.0f},
      }};
  std::vector<ContactReport> contact_reports = {
      {.from_agent_uuid = 12LL,
       .to_agent_uuid = kUuid,
       .test_result = {.probability = 0.5}},
      {.from_agent_uuid = 13LL,
       .to_agent_uuid = kUuid,
       .test_result = {.probability = 0.75}},
      {.from_agent_uuid = 15LL,
       .to_agent_uuid = kUuid,
       .test_result = {.probability = 0.95}},
  };
  auto outcomes = OutcomesFromContacts(kUuid, contacts);

  MockTransmissionModel transmission_model;
  EXPECT_CALL(transmission_model,
              GetInfectionOutcome(testing::ElementsAre(&outcomes[0].exposure,
                                                       &outcomes[1].exposure)));

  auto risk_score = absl::make_unique<MockRiskScore>();
  EXPECT_CALL(*risk_score, ContactRetentionDuration())
      .WillRepeatedly(Return(absl::Hours(24 * 14)));
  EXPECT_CALL(*risk_score,
              AddHealthStateTransistion(HealthTransition{
                  absl::InfinitePast(), HealthState::SUSCEPTIBLE}));

  EXPECT_CALL(*risk_score, GetContactTracingPolicy(_))
      .WillOnce(Return(RiskScore::ContactTracingPolicy{}));

  // This is a key assertion that we pass on exposures from
  // ProcessInfectionOutcomes to the risk score.
  EXPECT_CALL(*risk_score, AddExposures(testing::ElementsAre(
                               &outcomes[0].exposure, &outcomes[1].exposure)));

  // This a key assertion that we do pass on contact reports to the
  // risk score.
  // Only the contact_report form agents 12 and 15 are reported because the
  // agent had no matching exposure for agent 13.
  EXPECT_CALL(*risk_score, AddExposureNotification(
                               contacts[0], contact_reports[0].test_result));
  EXPECT_CALL(*risk_score, AddExposureNotification(
                               contacts[1], contact_reports[2].test_result));

  auto agent = SEIRAgent::CreateSusceptible(
      kUuid, &transmission_model, std::move(transition_model),
      std::move(visit_generator), std::move(risk_score));

  Timestep timestep(absl::UnixEpoch(), absl::Hours(24));
  agent->ProcessInfectionOutcomes(timestep, outcomes);
  agent->UpdateContactReports(timestep, contact_reports,
                              contact_report_broker.get());
}

TEST(SEIRAgentTest, PositiveTest) {
  auto transition_model = absl::make_unique<MockTransitionModel>();
  auto visit_generator = absl::make_unique<MockVisitGenerator>();

  auto risk_score = absl::make_unique<MockRiskScore>();
  EXPECT_CALL(*risk_score, ContactRetentionDuration())
      .WillRepeatedly(Return(absl::Hours(24 * 14)));
  EXPECT_CALL(*risk_score, AddHealthStateTransistion(HealthTransition{
                               .time = absl::InfinitePast(),
                               .health_state = HealthState::SUSCEPTIBLE}));
  HealthTransition transition = {
      .time = absl::FromUnixSeconds(-1LL),
      .health_state = HealthState::INFECTIOUS,
  };
  EXPECT_CALL(*risk_score, AddHealthStateTransistion(transition));

  const int64 kUuid = 42LL;
  std::vector<Contact> contacts = {{
      .other_uuid = 314LL,
      .exposure = {.start_time = absl::FromUnixSeconds(0LL),
                   .duration = absl::Hours(1LL),
                   .infectivity = 0.0f},
  }};
  std::vector<InfectionOutcome> outcomes =
      OutcomesFromContacts(kUuid, contacts);
  EXPECT_CALL(*risk_score,
              AddExposures(testing::ElementsAre(&outcomes[0].exposure)));

  EXPECT_CALL(*transition_model, GetNextHealthTransition(transition))
      .WillOnce(
          Return(HealthTransition{.time = absl::InfiniteFuture(),
                                  .health_state = HealthState::RECOVERED}));
  EXPECT_CALL(*risk_score, GetContactTracingPolicy(_))
      .WillOnce(Return(RiskScore::ContactTracingPolicy{.send_report = true}));
  TestResult expected_result = {
      .time_requested = absl::UnixEpoch(),
      .time_received = absl::UnixEpoch(),
      .probability = 1.0,
  };
  EXPECT_CALL(*risk_score, GetTestResult(_)).WillOnce(Return(expected_result));

  MockTransmissionModel transmission_model;
  auto agent =
      SEIRAgent::Create(kUuid,
                        {.time = absl::FromUnixSeconds(-1LL),
                         .health_state = HealthState::INFECTIOUS},
                        &transmission_model, std::move(transition_model),
                        std::move(visit_generator), std::move(risk_score));

  auto contact_report_broker = absl::make_unique<MockBroker<ContactReport>>();
  const std::vector<ContactReport> expected_contact_reports{
      {.from_agent_uuid = kUuid,
       .to_agent_uuid = 314LL,
       .test_result = expected_result}};
  EXPECT_CALL(*contact_report_broker, Send(Eq(expected_contact_reports)))
      .Times(1);
  const Timestep timestep(absl::UnixEpoch(), absl::Hours(24));

  agent->ProcessInfectionOutcomes(timestep, outcomes);
  agent->UpdateContactReports(timestep, {}, contact_report_broker.get());
}

TEST(SEIRAgentTest, NegativeTestResult) {
  auto transition_model = absl::make_unique<MockTransitionModel>();
  auto visit_generator = absl::make_unique<MockVisitGenerator>();
  const int64 kUuid = 42LL;
  auto risk_score = absl::make_unique<MockRiskScore>();
  EXPECT_CALL(*risk_score, ContactRetentionDuration())
      .WillRepeatedly(Return(absl::Hours(24 * 14)));
  const TestResult contact_test_result{
      .time_requested = absl::FromUnixSeconds(0LL),
      .time_received = absl::FromUnixSeconds(129600LL),
      .probability = 1.0f};
  std::vector<ContactReport> contact_reports{
      {.from_agent_uuid = 314LL,
       .to_agent_uuid = kUuid,
       .test_result = contact_test_result}};
  std::vector<Contact> contacts{
      {.other_uuid = 314LL,
       .exposure = {.start_time = absl::FromUnixSeconds(0LL),
                    .duration = absl::Hours(1LL)}}};
  EXPECT_CALL(*risk_score, AddHealthStateTransistion(HealthTransition{
                               .time = absl::InfinitePast(),
                               .health_state = HealthState::SUSCEPTIBLE}));
  std::vector<InfectionOutcome> outcomes =
      OutcomesFromContacts(kUuid, contacts);
  EXPECT_CALL(*risk_score,
              AddExposures(testing::ElementsAre(&outcomes[0].exposure)));
  EXPECT_CALL(*risk_score,
              AddExposureNotification(contacts[0], contact_test_result));

  EXPECT_CALL(*risk_score, GetContactTracingPolicy(_))
      .WillOnce(Return(RiskScore::ContactTracingPolicy{.send_report = false}));

  MockTransmissionModel transmission_model;
  EXPECT_CALL(transmission_model,
              GetInfectionOutcome(testing::ElementsAre(&outcomes[0].exposure)));

  auto agent = SEIRAgent::CreateSusceptible(
      kUuid, &transmission_model, std::move(transition_model),
      std::move(visit_generator), std::move(risk_score));

  auto contact_report_broker = absl::make_unique<MockBroker<ContactReport>>();
  EXPECT_CALL(*contact_report_broker, Send(testing::_)).Times(0);
  const Timestep timestep(absl::UnixEpoch(), absl::Hours(24));
  agent->ProcessInfectionOutcomes(timestep, outcomes);
  agent->UpdateContactReports(timestep, contact_reports,
                              contact_report_broker.get());
}

TEST(SEIRAgentTest, UpdatesAndPrunesContacts) {
  const int64 kUuid = 42LL;
  auto transition_model = absl::make_unique<MockTransitionModel>();
  EXPECT_CALL(*transition_model, GetNextHealthTransition(_))
      .WillRepeatedly(
          Return(HealthTransition{.time = absl::InfiniteFuture(),
                                  .health_state = HealthState::RECOVERED}));
  auto visit_generator = absl::make_unique<MockVisitGenerator>();
  MockTransmissionModel transmission_model;

  auto risk_score = absl::make_unique<MockRiskScore>();
  EXPECT_CALL(*risk_score, ContactRetentionDuration)
      .WillRepeatedly(Return(absl::Hours(1LL)));
  EXPECT_CALL(*risk_score, GetContactTracingPolicy(_))
      .WillRepeatedly(
          Return(RiskScore::ContactTracingPolicy{.send_report = true}));
  HealthTransition initial_transition = {
      .time = absl::FromUnixSeconds(-1LL),
      .health_state = HealthState::INFECTIOUS};
  EXPECT_CALL(*risk_score, AddHealthStateTransistion(HealthTransition{
                               .time = absl::InfinitePast(),
                               .health_state = HealthState::SUSCEPTIBLE}));
  EXPECT_CALL(*risk_score, AddHealthStateTransistion(initial_transition));

  const TestResult expected_test_result{
      .time_requested = absl::FromUnixSeconds(0LL),
      .time_received = absl::FromUnixSeconds(43200LL),
      .probability = 1.0f};

  std::vector<Contact> contacts1{
      {.other_uuid = 314LL,
       .exposure = {.start_time = absl::FromUnixSeconds(43201LL),
                    .duration = absl::Hours(1LL)}},
      {.other_uuid = 272LL,
       .exposure = {.start_time = absl::FromUnixSeconds(43200LL),
                    .duration = absl::Hours(1LL)}}};
  auto outcomes1 = OutcomesFromContacts(kUuid, contacts1);
  EXPECT_CALL(*risk_score,
              AddExposures(testing::ElementsAre(&outcomes1[0].exposure,
                                                &outcomes1[1].exposure)));
  EXPECT_CALL(*risk_score, GetTestResult(_))
      .WillRepeatedly(Return(expected_test_result));

  std::vector<Contact> contacts2{
      {.other_uuid = 314LL,
       .exposure = {.start_time = absl::FromUnixSeconds(86400LL),
                    .duration = absl::Hours(1LL)}}};
  auto outcomes2 = OutcomesFromContacts(kUuid, contacts2);
  EXPECT_CALL(*risk_score,
              AddExposures(testing::ElementsAre(&outcomes2[0].exposure)));

  auto agent =
      SEIRAgent::Create(kUuid, initial_transition, &transmission_model,
                        std::move(transition_model), std::move(visit_generator),
                        std::move(risk_score));

  Timestep timestep(absl::UnixEpoch(), absl::Hours(24));
  agent->ProcessInfectionOutcomes(timestep, outcomes1);

  // The first UpdateContactReports call will send a report to both of the
  // contacts we've had.
  auto contact_report_broker = absl::make_unique<MockBroker<ContactReport>>();
  {
    const std::vector<ContactReport> expected_contact_reports{
        {.from_agent_uuid = kUuid,
         .to_agent_uuid = 314LL,
         .test_result = expected_test_result},
        {.from_agent_uuid = kUuid,
         .to_agent_uuid = 272LL,
         .test_result = expected_test_result}};
    EXPECT_CALL(
        *contact_report_broker,
        Send(testing::UnorderedElementsAreArray(expected_contact_reports)))
        .Times(1);
  }
  agent->UpdateContactReports(timestep, {}, contact_report_broker.get());

  timestep.Advance();
  agent->ProcessInfectionOutcomes(timestep, outcomes2);

  // The second UpdateContactReports call will send a report to only for the
  // new contact.
  {
    const std::vector<ContactReport> expected_contact_reports{
        {.from_agent_uuid = kUuid,
         .to_agent_uuid = 314LL,
         .test_result = expected_test_result}};
    EXPECT_CALL(
        *contact_report_broker,
        Send(testing::UnorderedElementsAreArray(expected_contact_reports)))
        .Times(1);
  }
  agent->UpdateContactReports(timestep, {}, contact_report_broker.get());
}

TEST(SEIRAgentTest, UpdateContactReportsRejectsWrongUuid) {
  auto transition_model = absl::make_unique<MockTransitionModel>();
  auto visit_generator = absl::make_unique<MockVisitGenerator>();
  MockTransmissionModel transmission_model;
  auto risk_score = NewNullRiskScore();
  const int64 kUuid = 42LL;
  const std::vector<ContactReport> contact_reports{
      {.from_agent_uuid = kUuid, .to_agent_uuid = kUuid + 1}};
  auto agent = SEIRAgent::CreateSusceptible(
      kUuid, &transmission_model, std::move(transition_model),
      std::move(visit_generator), std::move(risk_score));
  auto broker = absl::make_unique<MockBroker<ContactReport>>();
  Timestep timestep(absl::UnixEpoch(), absl::Hours(24));
  ASSERT_DEBUG_DEATH(
      agent->UpdateContactReports(timestep, contact_reports, broker.get()), "");
}

}  // namespace
}  // namespace abesim
