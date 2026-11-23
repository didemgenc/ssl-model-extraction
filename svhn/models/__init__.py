from models import encoder
from models import losses
from models import resnet
try:
    from models import ssl  # SimCLR pre-training only; needs sklearn, unused by the revision scripts
except ImportError:
    ssl = None
from models import wideresnet

REGISTERED_MODELS = {} if ssl is None else {
    'sim-clr': ssl.SimCLR,
    'eval': ssl.SSLEval,
    'semi-supervised-eval': ssl.SemiSupervisedEval,
}
