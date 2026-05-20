import argparse
import multiprocessing

from config.config import Config
from module import ModuleManager


ModuleManagerInstance = ModuleManager()


def mapping_func(data_dict, histroy_query_buffer, config: Config):
    (
        Med_HOModule,
        Med_NMModule,
        Med_KGRMModule,
        Med_DVMModule,
        Med_LRMModule,
    ) = ModuleManagerInstance.mapper()

    origin_data = data_dict['query']
    function_call_type = data_dict['function_call_type']

    if function_call_type == 'Med-HOM':
        return Med_HOModule.process(origin_data, data_dict['clear'], histroy_query_buffer)

    if function_call_type == 'Med-NM':
        ner_res, _ = Med_NMModule.process(origin_data)
        return ner_res, histroy_query_buffer

    if function_call_type == 'Med-KGRM':
        ner_res, sim_threshold = Med_NMModule.process(origin_data)
        prompt = Med_KGRMModule.fetchKGPath(ner_res, origin_data, sim_threshold)
        return prompt, histroy_query_buffer

    if function_call_type == 'Med-DVM':
        gold_triplets = data_dict.get('gold_triplets', [])
        hypothesis = data_dict.get('hypothesis', origin_data)
        return Med_DVMModule.process(gold_triplets, origin_data, hypothesis), histroy_query_buffer

    if function_call_type == 'Med-LRM':
        gold_triplets = data_dict.get('gold_triplets', [])
        entity_description = data_dict.get('entity_description', {})
        return Med_LRMModule.process(gold_triplets, entity_description, origin_data), histroy_query_buffer

    return "Unsupported Med-HoDV function_call_type", histroy_query_buffer


def do_socket(conn, addr, config):
    histroy_query_buffer = []
    data_dict = None
    try:
        data_dict = conn.recv()
        print(data_dict)
    except EOFError:
        pass
    except Exception as e:
        print('Socket Error', e)

    if data_dict is None:
        out = ""
    else:
        try:
            out, histroy_query_buffer = mapping_func(data_dict, histroy_query_buffer, config)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print("Execution Error:", e)
            out = data_dict['query']
            histroy_query_buffer = []

    try:
        conn.send(out)
    except Exception as e:
        print('Socket Error:', e)

    try:
        conn.close()
        print('Connection close.', addr)
    except Exception:
        print('close except')


def err_call_back(err):
    print(f'error: {str(err)}')


def run_server(host, port, config: Config):
    from multiprocessing.connection import Listener

    multiprocessing.set_start_method('spawn')
    server_sock = Listener((host, port))
    print("Server running...", host, port)
    pool = multiprocessing.Pool(config.configServer.POOL_NUM)

    while True:
        conn = server_sock.accept()
        addr = server_sock.last_accepted
        print('Accept new connection', addr)
        pool.apply_async(func=do_socket, args=(conn, addr, config,), error_callback=err_call_back)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=r'./config/Med_HoDV_example.json')
    args = parser.parse_args()

    config = Config(args)
    ModuleManagerInstance.setup(config)
    run_server(config.configServer.SERVER_HOST, config.configServer.SERVER_PORT, config)


if __name__ == '__main__':
    main()
