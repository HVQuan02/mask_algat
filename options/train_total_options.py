from .base_options import BaseOptions

class TrainTotalOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('model', nargs='*', help='trained model')
        parser.add_argument('-L', '--use_local', action='store_true', help='use pretrained local model or not')
        parser.add_argument('-G', '--use_global', action='store_true', help='use pretrained global model or not')
        parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
        parser.add_argument('--milestones', nargs="+", type=int, default=[110, 160], help='milestones of learning decay')
        parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs to train')
        parser.add_argument('--batch_size', type=int, default=32, help='batch size')
        parser.add_argument('--save_scores', action='store_true', help='save the output scores')
        parser.add_argument('--save_path', default='scores.txt', help='output path')
        parser.add_argument('--resume', default=None, help='checkpoint to resume training')
        parser.add_argument('--patience', type=int, default=20, help='patience of early stopping')
        parser.add_argument('--min_delta', type=float, default=0.1, help='min delta of early stopping')
        parser.add_argument('--stopping_threshold', type=float, default=99, help='val mAP stopping_threshold of early stopping')
        return parser