import argparse
import datetime
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

    qa_epoch = data_dict['clear']
    origin_data = data_dict['query']

    print("Med-HOM start")
    starttime = datetime.datetime.now()
    data, histroy_query_buffer = Med_HOModule.process(origin_data, qa_epoch, histroy_query_buffer)
    endtime = datetime.datetime.now()
    print("Med-HOM finished in :", (endtime - starttime).seconds, " seconds")
    starttime = endtime

    gold_triplets, entity_description = [], {}
    if config.configKG.KG_ENABLE:
        print("Med-NM start")
        ner_res, sim_threshold = Med_NMModule.process(data)
        endtime = datetime.datetime.now()
        print("Med-NM finished in :", (endtime - starttime).seconds, " seconds")
        starttime = endtime

        print("Med-KGRM start")
        gold_triplets, entity_list = Med_KGRMModule.processKnowledge(ner_res, sim_threshold)
        endtime = datetime.datetime.now()
        print("Med-KGRM finished in :", (endtime - starttime).seconds, " seconds")
        starttime = endtime

        print("Med-KGRM(description) start")
        entity_description = Med_KGRMModule.processDescription(ner_res, origin_data)
        endtime = datetime.datetime.now()
        print("Med-KGRM(description) finished in :", (endtime - starttime).seconds, " seconds")
        starttime = endtime

    print("Med-DVM start")
    gold_triplets = Med_DVMModule.process(gold_triplets, origin_data, data)
    endtime = datetime.datetime.now()
    print("Med-DVM finished in :", (endtime - starttime).seconds, " seconds")

    if 'now_query' in data_dict.keys():
        origin_data = data_dict['now_query']

    if not config.configServer.RETURN_PROMPT:
        return [gold_triplets, entity_description, origin_data], histroy_query_buffer

    print("Med-LRM start")
    RAG_prompt = Med_LRMModule.process(gold_triplets, entity_description, origin_data)
    return RAG_prompt, histroy_query_buffer


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
