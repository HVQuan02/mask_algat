from .base_options import BaseOptions

class TrainGlobalOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
        parser.add_argument('--milestones', nargs="+", type=int, default=[50, 100], help='milestones of learning decay')
        parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs to train')
        parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        parser.add_argument('--resume', default=None, help='checkpoint to resume training')
        parser.add_argument('--patience', type=int, default=20, help='patience of early stopping')
        parser.add_argument('--min_delta', type=float, default=1e-4, help='min delta of early stopping')
        parser.add_argument('--stopping_threshold', type=float, default=0.01, help='val loss stopping_threshold of early stopping')
        return parser