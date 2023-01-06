from eval import EvalConfig, EvalSession, ConfusionType

# session = EvalSession(EvalConfig.kume_hanko)
# session = EvalSession(EvalConfig.kume_name)
session = EvalSession(EvalConfig.kume_waku)
# session = EvalSession(EvalConfig.kume_hanko_retrain)
# session = EvalSession(EvalConfig.kk_custom_hanko_cascade)
# session = EvalSession(EvalConfig.kk_custom_hanko_name_cascade)
# session = EvalSession(EvalConfig.kk_mixed_hanko_name_cascade)
session.load_gt()
session.calc_dt()
# session.evaluate()
session.show_eval_results()
# session.crop_detections(
#     dumpDir="croppedDetections",
#     targets=[
#         # ConfusionType.TP_RESULT,
#         ConfusionType.FP_RESULT
#     ],
#     iouThresh=0.5,
#     overwrite=True
# )
# session.infer_new_annotations(newLimit=20)
