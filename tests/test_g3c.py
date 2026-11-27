"""Known-answer checks for observed geometry, fallback and sealed context."""
import importlib.util
from pathlib import Path
import sys
import unittest

AVAILABLE=all(importlib.util.find_spec(x) for x in ("numpy","scipy","matplotlib"))
if AVAILABLE:
    import duckdb
    import numpy as np
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
    from g3a_policy import make_splits
    from g3c_construct import geometry, perturbation_bound, verify_bound
    from g3c_context import components,read_context
    from g3c_policy import predict


@unittest.skipUnless(AVAILABLE,"G3C requires the optional scientific packages")
class G3CTest(unittest.TestCase):
    def test_observed_geometry_bound(self):
        nodes=np.array([[0.,0.],[3.,0.],[3.,4.],[3.,5.]])
        c,l=geometry(nodes)
        self.assertAlmostEqual(l,8)
        self.assertAlmostEqual(c,np.sqrt(34)/8)
        self.assertAlmostEqual(perturbation_bound(c,l,0),0)
        with self.assertRaises(ValueError):
            perturbation_bound(c,l,l/6+1e-5)
        self.assertEqual(verify_bound()["cases"],2000)

    def test_weighted_variance_identity(self):
        v,g=components(np.array([0.,2.,1.,3.]),np.array([1.,1.,2.,2.]),{"p":np.array([0,0,1,1])})
        self.assertAlmostEqual(g["p"]["within"],1)
        self.assertAlmostEqual(g["p"]["between"],2/9)
        self.assertAlmostEqual(v,11/9)

    def test_full_policy_fallbacks(self):
        i=np.arange(12,dtype=float)
        pos=(i%2)[:,None]
        y=i*.02+1.5
        forty=4+i/20
        forty[0]=np.nan
        tech=(i%3!=0).astype(float)
        short=1.5+i/30
        short[:2]=np.nan
        cobs=i/12
        cobs[2]=np.nan
        fold=make_splits(y,False)
        z=predict((y,pos,forty,tech,short,cobs),fold,np.ones(len(i),dtype=int),False,(True,True))
        np.testing.assert_allclose(z[:,0,0],z[:,0,1])
        np.testing.assert_allclose(z[:,:2,1],z[:,:2,3])
        np.testing.assert_allclose(z[:,[0,2],4],z[:,[0,2],5])
        self.assertEqual(z.shape,(3,12,6))

    def test_retrospective_context_seals_later_cohorts_and_keeps_unknown(self):
        with duckdb.connect() as db:
            db.execute("""CREATE TABLE players(nfl_id INT,nfl_position VARCHAR,draft_year INT);
            INSERT INTO players VALUES (1,'WR',2023),(2,'WR',2024),(3,'TE',2025);
            CREATE TABLE games(game_id INT,season INT,season_type VARCHAR,home_team_abbr VARCHAR,visitor_team_abbr VARCHAR);
            INSERT INTO games VALUES (1,2023,'REG','A','B'),(2,2024,'REG','A','B');
            CREATE TABLE player_play AS SELECT i AS nfl_id,1 AS game_id, 1 AS play_id,
               'OUT' AS route_ran,2.5::DOUBLE AS separation_at_pass_forward,'A' AS team_abbr,
               1 AS down, 10 AS yards_to_go,40 AS yardline_number,'I' AS offense_formation,
               '2x1' AS receiver_alignment, 7 AS pass_length,'COVER_3' AS team_coverage_type,
               false AS target, NULL AS dropback_duration FROM range(1,4) AS a(i);
            INSERT INTO player_play SELECT nfl_id,2,2,route_ran,999,team_abbr,down,yards_to_go, yardline_number,
                  offense_formation,receiver_alignment,pass_length,team_coverage_type,target,dropback_duration
                  FROM player_play WHERE nfl_id=1;
            CREATE TABLE game_tracking_2023(nfl_id INT,game_id INT,play_id INT,event VARCHAR,time TIMESTAMP);
            INSERT INTO game_tracking_2023 VALUES (1,1,1,'ball_snap','2023-09-01 12:00:00'),
              (1,1,1,'pass_forward','2023-09-01 12:00:02'),(1,1,1,'pass_forward','2023-09-01 12:00:03');""")
            a=read_context(db)
            self.assertEqual(len(a),1)
            self.assertEqual(a[0]["player"],1)
            self.assertEqual(a[0]["opponent"],"B")
            self.assertEqual(a[0]["throw_count"],2)
            self.assertIsNone(a[0]["seconds"])
