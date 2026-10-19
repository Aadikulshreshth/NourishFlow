import unittest

from rescue_route.agents import run_rescue_workflow
from rescue_route.data import demo_scenario, sample_donations, sample_recipients
from rescue_route.optimizer import validate_plan


class NourishFlowWorkflowTests(unittest.TestCase):
    def test_baseline_produces_a_valid_plan(self):
        donations, recipients, _ = demo_scenario("Baseline")
        result = run_rescue_workflow(donations, recipients)
        self.assertTrue(result["success"])
        self.assertIn(result["status"], {"completed", "completed_after_retry"})
        self.assertFalse(result["plan"].empty)
        self.assertTrue(validate_plan(result["plan"], donations, recipients).valid)

    def test_supply_and_demand_constraints_are_respected(self):
        donations, recipients, _ = demo_scenario("Baseline")
        result = run_rescue_workflow(donations, recipients)
        plan = result["plan"]
        for row in donations.itertuples(index=False):
            used = int(plan.loc[plan["donation_id"] == row.donation_id, "allocated_kg"].sum())
            self.assertLessEqual(used, row.quantity_kg)
        for row in recipients.itertuples(index=False):
            delivered = int(plan.loc[plan["recipient_id"] == row.recipient_id, "allocated_kg"].sum())
            self.assertLessEqual(delivered, min(row.demand_kg, row.storage_capacity_kg))

    def test_invalid_schema_is_rejected(self):
        donations, recipients, _ = demo_scenario("Invalid input")
        result = run_rescue_workflow(donations, recipients)
        self.assertEqual(result["status"], "rejected")
        self.assertFalse(result["input_validation"].valid)

    def test_critic_rejects_fault_and_correction_retry_passes(self):
        donations, recipients, _ = demo_scenario("Baseline")
        result = run_rescue_workflow(donations, recipients, inject_critic_failure=True)
        self.assertEqual(result["status"], "completed_after_retry")
        self.assertTrue(result["critic_validation"].valid)
        self.assertTrue(any(log.stage == "Correction Agent" for log in result["logs"]))
        self.assertTrue(any(log.status == "FAIL" and log.stage == "Critic Agent" for log in result["logs"]))

    def test_disruption_scenario_uses_remaining_donations(self):
        baseline_d, recipients, _ = demo_scenario("Baseline")
        disrupted_d, disrupted_r, _ = demo_scenario("Donation disruption")
        baseline = run_rescue_workflow(baseline_d, recipients)
        disrupted = run_rescue_workflow(disrupted_d, disrupted_r)
        self.assertNotIn("D03", set(disrupted_d["donation_id"]))
        self.assertTrue(disrupted["success"])
        self.assertLessEqual(disrupted["summary"]["total_allocated_kg"], baseline["summary"]["total_allocated_kg"])

    def test_shortage_leaves_some_demand_unmet(self):
        donations, recipients, _ = demo_scenario("Supply shortage")
        result = run_rescue_workflow(donations, recipients)
        self.assertTrue(result["success"])
        self.assertLess(result["summary"]["total_allocated_kg"], result["summary"]["demand_capacity_kg"])


if __name__ == "__main__":
    unittest.main()
