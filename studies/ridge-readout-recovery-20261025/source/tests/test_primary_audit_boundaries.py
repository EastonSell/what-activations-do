"""Independent-review generated correctness fixture, not historical observations."""
import unittest
import numpy as np
import audit
from protocol_v2 import readout

class PrimaryAuditBoundaryTests(unittest.TestCase):
    def test_strict_margin_uses_frozen_float_operation_order(self):
        losses=[881,1260,1741,1683,571,1103,1218,133,990,22,1772,333,696,775,1257,97,293,1489,891,693,1217,643,1283,1592,1342,507,56,1276,1026,349,1030,197,419,593,432,970,1503,1087,658,1281,514,1458,882,1787,1338,1134,946,342,1548]
        a=np.ones((5,360), dtype=bool); b=a.copy(); a.flat[losses]=False
        expected=readout.primary(a,b); actual=audit.primary(a,b)
        self.assertEqual(expected['decision'],'two_point_margin_not_met_by_sensitivity_rule')
        self.assertEqual(actual['decision'],expected['decision'])
        self.assertEqual(actual['case_resampling_interval95_pp'],expected['case_resampling_interval95_pp'])

if __name__=='__main__': unittest.main()
