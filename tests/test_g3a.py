"""Invented inputs for measurement, sealed cohort and partial review checks."""
import importlib.util
from pathlib import Path
import sys
import unittest

AVAILABLE=all(importlib.util.find_spec(x) is not None for x in ("numpy","scipy","matplotlib"))
if AVAILABLE:
    import duckdb
    import numpy as np
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
    from g3a_annotation import agreement
    from g3a_measurement import describe
    from g3a_policy import fold_policy,losses,read_outcomes


@unittest.skipUnless(AVAILABLE,"G3A measurement requires optional scientific packages")
class G3ATest(unittest.TestCase):
    def test_noiseless_threshold_and_length(self):
        t=np.arange(41)*.1
        for method in ("raw_interior","quadratic"):
            self.assertFalse(describe(t,np.c_[0*t,0*t],method)["accepted"])
            self.assertFalse(describe(t,np.c_[.6*t,0*t],method)["accepted"])
            a=describe(t,np.c_[4*t,0*t],method)
            self.assertTrue(a["accepted"])
            self.assertAlmostEqual(a["length"],14.4)
            self.assertAlmostEqual(a["chord_ratio"],1)
            self.assertAlmostEqual(a["velocity_contrast"],0)
            self.assertFalse(describe(np.delete(t,10),np.c_[4*np.delete(t,10),np.delete(t,10)*0],method)["accepted"])

    def test_query_seals_later_cohorts_and_missing_targets(self):
        with duckdb.connect() as db:
            db.execute("""CREATE TABLE players(nfl_id INT,nfl_position VARCHAR,draft_year INT);
                          INSERT INTO players VALUES (1,'WR',2023),(2,'TE',2023),(3,'WR',2024),(4,'WR',2025);
                          CREATE TABLE games(game_id INT,season INT,season_type VARCHAR);
                          INSERT INTO games VALUES (1,2023,'REG'),(2,2024,'REG');
                          CREATE TABLE combine_results(nfl_id INT,forty DOUBLE);
                          INSERT INTO combine_results VALUES(1,4.5);
                          CREATE TABLE player_play AS SELECT
                            1 AS nfl_id,1 AS game_id,'OUT' AS route_ran,2.0 AS separation_at_pass_forward
                            FROM range(25)
                          UNION ALL SELECT 2,1,'OUT',NULL FROM range(25)
                          UNION ALL SELECT 1,2,'OUT',1000 FROM range(25)
                          UNION ALL SELECT 3,1,'OUT',3000 FROM range(25)
                          UNION ALL SELECT 4,1,'OUT',4000 FROM range(25)""")
            self.assertEqual(read_outcomes(db),[(1,"WR",4.5,25,2.0),(2,"TE",None,0,None)])

    def test_nested_fallback_and_sealed_folds(self):
        y=np.arange(12,dtype=float)
        test=np.array([10,11])
        rows=np.arange(10)
        pos=(y%2)[:,None]
        forty=4+y/10
        forty[-1]=np.nan
        g=np.r_[np.zeros(6),np.ones(6)]
        x=y[:,None].copy()
        x[-2]=np.nan
        weights=np.ones(len(y),dtype=int)
        pred,_=fold_policy((y,pos,forty,g,x),rows,test,weights,False)
        y2=y.copy()
        y2[test]+=1000
        check,_=fold_policy((y2,pos,forty,g,x),rows,test,weights,False)
        np.testing.assert_allclose(pred,check)
        self.assertEqual(pred[1,0],pred[1,1])
        np.testing.assert_allclose(pred[:,2],pred[:,3])
        self.assertAlmostEqual(losses(y[test],pred[None],False)["mse"][0,0,0],(pred[0,0]-10)**2)

    def test_partial_and_same_player_cluster(self):
        a=dict(onset_s=1.,bend_start_s=1.1,bend_end_s=None,bend_center_s=1.3,
               exit_s=None,index=1,direction="left",confidence="low")
        b=a|dict(onset_s=1.05,bend_center_s=1.4,exit_s=1.7)
        left={"A":[("events",a)],"B":[("events",a)]}
        right={"A":[("events",b)],"B":[("no_event",None)],"C":[("unjudgeable",None)]}
        r=agreement(left,right,{"A":"g1","B":"g1","C":"g2"})
        self.assertEqual(r["both_submitted"],2)
        self.assertEqual(r["clusters"],1)
        self.assertEqual(r["estimates"]["complete_f1"],0)
        self.assertAlmostEqual(r["estimates"]["expanded_agreement"],2/3)
        self.assertEqual(r["estimates"]["partial_exit_s_pairs"],0)
        self.assertAlmostEqual(r["estimates"]["partial_bend_center_s_mae"],.1)
