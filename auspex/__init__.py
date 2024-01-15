import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'lib')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Helcaraxe_models')))
#import boost_adaptbx.boost.python as bp
#bp.import_ext('dxtbx_model_ext')

def run():
    import Parser

if __name__ == '__main__':
    run()


