import time
import os
import sys


class Logger(object):

    def __init__(self, stream=sys.stdout, model='model', loss='loss', attack='attack'):
        output_dir = "/tmp/Adversarial_Examples/train/log"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        log_name = '{}.log'.format(time.strftime(model + '-' + loss + '-' + attack + '-%Y-%m-%d-%H-%M'))
        filename = os.path.join(output_dir, log_name)
        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
