import unittest
from unittest.mock import patch

import analytics
import alerts


# Same tuple structure returned by database.get_all_records():
# (id, timestamp, student_id, engagement_score, status,
#  registered_student_id, session_id, orientation)

TEST_RECORDS = [
    # Newest records first, matching database.py ORDER BY id DESC
    (6, "2026-09-23T10:05:00", 20, 85, "Engaged", 2, 1, "Forward"),
    (5, "2026-09-23T10:04:00", 14, 75, "Engaged", 2, 1, "Forward"),
    (4, "2026-09-23T10:03:00", 9, 80, "Engaged", 2, 1, "Forward"),

    # Same registered student, but different temporary track IDs
    (3, "2026-09-23T10:02:00", 15, 40, "Low Engagement", 3, 1, "Head Right"),
    (2, "2026-09-23T10:01:00", 12, 45, "Low Engagement", 3, 1, "Looking Down"),
    (1, "2026-09-23T10:00:00", 8, 50, "Low Engagement", 3, 1, "Head Left"),
]


class TestAnalytics(unittest.TestCase):

    @patch("analytics.get_all_records", return_value=TEST_RECORDS)
    def test_student_average(self, mock_records):
        self.assertEqual(analytics.get_student_average(3), 45.0)
        self.assertEqual(analytics.get_student_average(2), 80.0)

    @patch("analytics.get_all_records", return_value=TEST_RECORDS)
    def test_session_average(self, mock_records):
        self.assertEqual(analytics.get_session_average(1), 62.5)

    @patch("analytics.get_all_records", return_value=TEST_RECORDS)
    def test_low_engagement_count(self, mock_records):
        self.assertEqual(analytics.get_low_engagement_count(), 3)

    @patch("analytics.get_all_records", return_value=TEST_RECORDS)
    def test_student_comparison(self, mock_records):
        result = analytics.get_student_comparison()

        self.assertEqual(result[2], 80.0)
        self.assertEqual(result[3], 45.0)

    @patch("analytics.get_all_records", return_value=TEST_RECORDS)
    def test_session_comparison(self, mock_records):
        result = analytics.get_session_comparison()

        self.assertEqual(result[1], 62.5)

    @patch("analytics.get_all_records", return_value=TEST_RECORDS)
    def test_daily_engagement_averages(self, mock_records):
        result = analytics.get_daily_engagement_averages()

        self.assertEqual(result["2026-09-23"], 62.5)

    @patch("analytics.get_all_records", return_value=TEST_RECORDS)
    def test_weekly_engagement_averages(self, mock_records):
        result = analytics.get_weekly_engagement_averages()

        self.assertEqual(result["2026-W39"], 62.5)

    @patch("analytics.get_all_records", return_value=[])
    def test_empty_data(self, mock_records):
        result = analytics.get_analytics()

        self.assertEqual(result["average_engagement"], 0)
        self.assertEqual(result["total_records"], 0)
        self.assertEqual(result["low_engagement_count"], 0)
        self.assertEqual(result["student_comparison"], {})
        self.assertEqual(result["session_comparison"], {})
        self.assertEqual(result["daily_averages"], {})
        self.assertEqual(result["weekly_averages"], {})


class TestAlerts(unittest.TestCase):

    def test_score_below_threshold_alerts(self):
        result = alerts.check_alert("Track-1", 59)

        self.assertTrue(result["alert"])

    def test_score_at_threshold_does_not_alert(self):
        result = alerts.check_alert("Track-1", 60)

        self.assertFalse(result["alert"])

    def test_score_above_threshold_does_not_alert(self):
        result = alerts.check_alert("Track-1", 80)

        self.assertFalse(result["alert"])

    @patch("alerts.get_all_records", return_value=TEST_RECORDS)
    def test_sustained_low_engagement(self, mock_records):
        result = alerts.check_sustained_low_engagement(3)

        self.assertTrue(result["alert"])

    @patch("alerts.get_all_records", return_value=TEST_RECORDS)
    def test_engaged_student_not_flagged(self, mock_records):
        result = alerts.check_sustained_low_engagement(2)

        self.assertFalse(result["alert"])

    @patch("alerts.get_all_records", return_value=[])
    def test_empty_data_no_sustained_alert(self, mock_records):
        result = alerts.check_sustained_low_engagement(3)

        self.assertFalse(result["alert"])


if __name__ == "__main__":
    unittest.main()