"""Known answers for multiscale operators and Combine-only reference queries."""
import importlib.util
from pathlib import Path
import sys
import unittest

AVAILABLE=all(importlib.util.find_spec(x) for x in ("numpy","scipy","matplotlib"))
if AVAILABLE:
    import duckdb
    import numpy as np
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
    from g3ar1_measure import METHODS,FEATURES,features,native
    from g3ar1_reference import crossing, read


@unittest.skipUnless(AVAILABLE,"G3A-R1 uses optional scientific packages")
class G3AR1Test(unittest.TestCase):
    def test_constant_physical_speed_at_all_spans(self):
        t=.05+np.arange(41)*.1
        p=np.c_[.6*t,0*t]
        for method in METHODS:
            _,v=native(t,p,method)
            np.testing.assert_allclose(v,.6,atol=1e-10)
            self.assertTrue(np.all(v<1.))
            self.assertTrue(np.all(v>.3))
        self.assertIsNone(features(t,p)[FEATURES[0]])  # Only 1.8 yd support.
        self.assertIsNone(features(t,p)[FEATURES[1]])

    def test_reverse_chord_and_geometric_construct(self):
        t=np.arange(41)*.1
        p=np.c_[4*np.minimum(t,1.5)-4*np.maximum(t-1.5,0),0*t]
        q,v=native(t,p,"d10")
        self.assertAlmostEqual(v[np.argmin(abs(q-1.5))],0)
        self.assertGreater(np.max(v),3.99)
        self.assertAlmostEqual(features(t,p)[FEATURES[0]],0)
        self.assertAlmostEqual(features(t,p)[FEATURES[1]],0)
        self.assertEqual(native(np.delete(t,10),np.delete(p,10,0),"d10")[0].size,0)

    def test_relative_crossings_not_extrapolated(self):
        t=np.array([0.,1.,2.])
        x=np.array([0.,8.,12.])
        self.assertAlmostEqual(crossing(t,x,10),1.5)
        self.assertIsNone(crossing(t,x,40))

    def test_reference_query_is_2023_only(self):
        with duckdb.connect() as d:
            d.execute("""
                CREATE TABLE players(nfl_id INT,nfl_position VARCHAR,draft_year INT);
                INSERT INTO players VALUES(1,'WR',2023),(2,'TE',2024),(3,'WR',2025);
                CREATE TABLE combine_results(nfl_id INT,draft_year INT,forty DOUBLE,
                  ten_yd_split DOUBLE,vertical DOUBLE,broad_jump DOUBLE,short_shuttle DOUBLE,three_cone DOUBLE);
                INSERT INTO combine_results(nfl_id,draft_year,forty)
                  VALUES(1,2023,4.5),(1,2024,900),(2,2024,901),(3,2025,902);
                CREATE TABLE combine_tracking AS
                    SELECT nfl_id,draft_year,'e' AS event_id,'FORTY_YARD_DASH' AS drill_name,
                      'PLAYER' AS entity_type,1 AS attempt,TIMESTAMP '2023-01-01' AS time,
                      0.0 AS x,0.0 AS y,0.0 AS s,0.0 AS a,0.0 AS dir,0.0 AS dis
                    FROM players;
            """)
            tracks,fields=read(d)
            self.assertEqual(len(tracks),1)
            self.assertEqual(tracks[0][0],1)
            self.assertEqual(fields,[(1,"WR",4.5,None,None,None,None,None)])
