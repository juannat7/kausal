function quickpatch(nn,FxPred,FxUnc,colorstr)
patch([nn;flipud(nn)],[FxPred+(FxUnc);flipud(FxPred-(FxUnc))],colorstr,'FaceAlpha',0.1,'EdgeColor','none')
end

