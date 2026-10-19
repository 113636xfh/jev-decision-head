"""Head first; uncertain or invalid predictions fall back to the SAME backbone's decode."""
import math
from dataclasses import dataclass
from domain import validate_distribution

@dataclass(frozen=True)
class Policy:
    min_confidence:float=.8
    min_margin:float=.2
    max_normalized_entropy:float=.55
    force_decode:bool=False
    def __post_init__(self):
        if not all(0<=x<=1 for x in (self.min_confidence,self.min_margin,self.max_normalized_entropy)):
            raise ValueError('policy thresholds must be in [0,1]')
    def assess(self,probs):
        validate_distribution(probs,len(probs))
        if len(probs)<2:raise ValueError('need two or more candidates')
        ranked=sorted(probs,reverse=True);conf=ranked[0];margin=ranked[0]-ranked[1]
        entropy=-sum(p*math.log(p) for p in probs if p>0)/math.log(len(probs))
        accepted=not self.force_decode and conf>=self.min_confidence and margin>=self.min_margin and entropy<=self.max_normalized_entropy
        return {'accepted':accepted,'confidence':conf,'margin':margin,'normalized_entropy':entropy}

def decide(item,head_predictor,decode_predictor,policy):
    probs=None
    try:
        probs=head_predictor(item);assessment=policy.assess(probs)
        if len(probs)!=len(item['options']):raise ValueError('candidate count mismatch')
        if assessment['accepted']:
            i=max(range(len(probs)),key=lambda j:probs[j])
            return {'route':'head','choice':item['options'][i],'head_probabilities':probs,'assessment':assessment}
        reason='uncertain'
    except Exception as exc:assessment={'head_error':type(exc).__name__+': '+str(exc)};reason='invalid_head'
    try:
        decoded=decode_predictor(item)
        if decoded.get('choice') not in item['options']:raise ValueError('decode gave no valid answer')
        return {'route':'decode','choice':decoded['choice'],'fallback_reason':reason,'assessment':assessment,'head_probabilities':probs,'decode':decoded}
    except Exception as exc:
        return {'route':'failed','choice':None,'fallback_reason':reason,'assessment':assessment,'head_probabilities':probs,'error':type(exc).__name__+': '+str(exc)}

def summarize(rows):
    # Denominator includes failed head/decode; only independently labeled rows score accuracy.
    totals={'requests':len(rows),'head':0,'decode':0,'failed':0,'independently_labeled':0,'correct':0}
    for row in rows:
        totals[row['route']]+=1
        if row.get('independent_gold') is not None:
            totals['independently_labeled']+=1
            totals['correct']+=int(row.get('choice')==row['independent_gold'])
    n=totals['requests'];ng=totals['independently_labeled'];valid=n-totals['failed']
    totals.update(head_coverage=totals['head']/n if n else None,decode_rate=totals['decode']/n if n else None,
                  valid_answer_rate=valid/n if n else None,accuracy=totals['correct']/ng if ng else None)
    labeled_valid=sum(r['route']!='failed' and r.get('independent_gold') is not None for r in rows)
    totals['conditional_accuracy']=totals['correct']/labeled_valid if labeled_valid else None
    return totals
