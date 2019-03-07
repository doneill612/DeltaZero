import asyncio
import os
import platform

import chess.engine as engine
import numpy as np

from core.dzlogging import Logger

logger = Logger.get_logger('Stockfish')

class Stockfish(object):
    '''Utility class for performing board evaluations.'''
    
    class _Stockfish(object):
        '''Singleton object'''
        def __init__(self):
            self.t = None
            self.e = None
            asyncio.set_event_loop_policy(engine.EventLoopPolicy())
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        def start(self):
            
            async def establish():
                current_os = platform.system()
                ep = os.path.join(os.path.pardir, 'data',
                                  f'stockfish-10-{"win" if current_os == "Windows" else "linux"}',
                                  f'{current_os}',
                                  f'stockfish_10_x64{".exe" if current_os == "Windows" else ""}')
                t, e = await engine.popen_uci(ep)
                return t, e

            self.t, self.e = self.loop.run_until_complete(asyncio.gather(establish()))[0]
            logger.info('Engine startup success.')

        def evaluate(self, env):
            async def get_score(e, env):
                board = env.board
                evaltime = 0.005
                info = await e.analyse(board, limit=engine.Limit(time=evaltime))
                score = info['score'].white().score(mate_score=10000)
                if score > 1.5:
                    return 1
                elif score < -1.5:
                    return -1
                else:
                    return 0
                
            score = self.loop.run_until_complete(asyncio.gather(get_score(self.e, env)))[0]
            return score

        def close(self):
            async def end(t, e):
                await e.quit()
                t.close()

            self.loop.run_until_complete(asyncio.gather(end(self.t, self.e)))
            logger.info('Engine shutdown success.')
                
        
    ENGINE = _Stockfish()

    @staticmethod
    def evalutate(env):
        return ENGINE.evaluate(env)

    @staticmethod
    def close():
        ENGINE.close()

    @staticmethod
    def start():
        ENGINE.start()
