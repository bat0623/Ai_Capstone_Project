import json
import pandas as pd
import textwrap

class JsonlWriter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = open(file_path, 'a', encoding='utf-8')

    def new_file(self):
        self.file.close()
        with open(self.file_path, 'w', encoding='utf-8'):
            pass
        self.file = open(self.file_path, 'a', encoding='utf-8')

    def add_jsonl(self, data):
        conv_json=json.dumps(data, ensure_ascii=False, indent=4)
        self.file.write(conv_json+'\n')

    def close(self):
        if not self.file.closed:
            self.file.close()
        return self.file_path

    def __del__(self):
        self.close()



class ConversationMaker:
    def __init__(self, player_dict, npc_dict, conversation_formated_dict):
        self.player_df = pd.DataFrame(player_dict)
        self.npc_df = pd.DataFrame(npc_dict)
        self.conversation_dict = conversation_formated_dict

    def conversation_generator_jsonl(self, player_code, npc_code, situation_name=None):
        player_info = self.player_df[self.player_df["code"] == player_code].iloc[0]
        npc_info = self.npc_df[self.npc_df["code"] == npc_code].iloc[0]

        if situation_name is not None:
            yield from self.__format_conversation(player_info, npc_info, situation_name)
        else:
            for conversation_key in self.conversation_dict:
                yield from self.__format_conversation(player_info, npc_info, conversation_key)


    def __get_situation_description(self, situation_name):
        #situation = self.conversation_dict[situation_name]
        return "병신"

    def __format_conversation(self, player_info, npc_info, situation_name):

        situation = self.conversation_dict[situation_name]
        question_list = situation["questions"]
        answer_list = situation["answers"]

        for question in question_list:
            for answer in answer_list:
                formated_question = question.format(p_name=npc_info["name"],
                                                    p_city=npc_info["city"],
                                                    p_age=npc_info["age"],
                                                    p_gender=npc_info["gender"],
                                                    p_job=npc_info["job"],
                                                    p_social_status=npc_info["social_status"],
                                                    p_era=npc_info["era"]
                                                    )
                formated_answer = answer.format(n_name=npc_info["name"],
                                                n_city=npc_info["city"],
                                                n_age=npc_info["age"],
                                                n_gender=npc_info["gender"],
                                                n_job=npc_info["job"],
                                                n_social_status=npc_info["social_status"],
                                                n_era=npc_info["era"]
                                                )

                instruction = "배경: {n_era}  도시: {n_city}    플레이어: {p_name}({p_job}, {p_social_status})  NPC: {n_name}({n_job}, {n_social_status}, {n_relation}))"
                instruction+=f" 상황: {situation['description']}"
                instruction = instruction.format(n_name=npc_info["name"],
                                                 n_city=npc_info["city"],
                                                 n_age=npc_info["age"],
                                                 n_gender=npc_info["gender"],
                                                 n_job=npc_info["job"],
                                                 n_social_status=npc_info["social_status"],
                                                 n_era=npc_info["era"],
                                                 n_relation=npc_info["relation"],

                                                 p_name=player_info["name"],
                                                 p_gender=player_info["gender"],
                                                 p_job=player_info["job"],
                                                 p_social_status=player_info["social_status"],
                                                 )

                conv_jsonl = {
                    "instruction": instruction,
                    "input": formated_question,
                    "output": formated_answer
                }
                yield conv_jsonl


if __name__ == "__main__":
    index=0
    player_dict = json.load(open("sample_instruction_info.json"))["players"]
    npc_dict = json.load(open("npcs_info.json"))
    conversation_formated_dict = json.load(open("conversation_sample.json"))

    conversation_maker = ConversationMaker(player_dict, npc_dict, conversation_formated_dict)
    jsonl_writer = JsonlWriter("conversation_backup.jsonl")
    jsonl_writer.new_file()

    try:
        for player in player_dict:
            player_code = player["code"]
            for npc in npc_dict:
                npc_code = npc["code"]
                conv_jsonl_gen = conversation_maker.conversation_generator_jsonl(player_code, npc_code)
                for conv in conv_jsonl_gen:
                    print(index,"\t",json.dumps(conv, ensure_ascii=False, indent=4))
                    jsonl_writer.add_jsonl(conv)
                    index+=1


    except Exception as e:
        print(e)

    finally:
        path=jsonl_writer.close()
        print(f"{path} 저장 완료")
