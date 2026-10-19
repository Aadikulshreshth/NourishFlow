import unittest

from rescue_route.agents import run_rescue_workflow
from rescue_route.data import demo_scenario, sample_donations, sample_recipients
from rescue_route.optimizer import optimize_allocations, validate_plan
from rescue_route.routing import default_fleet, optimize_vehicle_routes, validate_route_plan


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

    def test_vehicle_routes_cover_every_allocation_and_pass_route_critic(self):
        donations, recipients, _ = demo_scenario("Baseline")
        allocation, _ = optimize_allocations(donations, recipients)
        routes = optimize_vehicle_routes(allocation, donations, recipients)
        validation = validate_route_plan(routes, allocation)
        self.assertTrue(routes["success"], routes.get("metadata", {}).get("failure_reason"))
        self.assertTrue(validation.valid, validation.errors)
        self.assertEqual(routes["summary"]["total_delivered_kg"], int(allocation["allocated_kg"].sum()))
        self.assertLessEqual(routes["summary"]["max_vehicle_load_kg"], 250)
        self.assertEqual(len(routes["stops"].query("stop_type == 'Delivery'")), len(allocation))

    def test_vehicle_breakdown_replans_with_remaining_fleet(self):
        donations, recipients, _ = demo_scenario("Vehicle breakdown / replan")
        result = run_rescue_workflow(donations, recipients, simulate_vehicle_breakdown=True)
        self.assertTrue(result["routing_replan"]["requested"])
        self.assertTrue(result["routing_replan"]["success"], result["routing_replan"]["message"])
        self.assertIsNotNone(result["routing_replan"]["broken_vehicle_id"])
        self.assertTrue(result["initial_routing"]["success"])
        self.assertTrue(result["routing"]["success"])
        self.assertEqual(result["initial_routing"]["summary"]["vehicles_used"], 3)
        self.assertEqual(result["routing"]["summary"]["vehicles_available"], 2)
        self.assertEqual(result["routing"]["summary"]["vehicles_used"], 2)
        original_vehicle_ids = set(result["initial_routing"]["stops"]["vehicle_id"])
        revised_vehicle_ids = set(result["routing"]["stops"]["vehicle_id"])
        self.assertEqual(len(original_vehicle_ids), 3)
        self.assertEqual(len(revised_vehicle_ids), 2)
        self.assertNotIn(result["routing_replan"]["broken_vehicle_id"], revised_vehicle_ids)
        original_pickups = result["initial_routing"]["stops"].query("stop_type == 'Pickup'")
        revised_pickups = result["routing"]["stops"].query("stop_type == 'Pickup'")
        original_assignments = {
            (row.donation_id, row.recipient_id, int(row.quantity_kg)): row.vehicle_id
            for row in original_pickups.itertuples(index=False)
        }
        revised_assignments = {
            (row.donation_id, row.recipient_id, int(row.quantity_kg)): row.vehicle_id
            for row in revised_pickups.itertuples(index=False)
        }
        self.assertTrue(any(original_assignments[k] != revised_assignments[k] for k in original_assignments))
        self.assertEqual(result["routing"]["summary"]["total_delivered_kg"], 485)

    def test_routing_rejects_shipment_larger_than_vehicle_capacity(self):
        donations, recipients, _ = demo_scenario("Baseline")
        allocation, _ = optimize_allocations(donations, recipients)
        routes = optimize_vehicle_routes(allocation, donations, recipients, vehicle_specs=default_fleet(3, 100))
        self.assertFalse(routes["success"])
        self.assertIn("exceeds every available vehicle capacity", routes["metadata"]["failure_reason"])


if __name__ == "__main__":
    unittest.main()
