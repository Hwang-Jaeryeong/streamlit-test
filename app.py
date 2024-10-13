import streamlit as st
from llama_index.core.llms import ChatMessage
import logging
import time
from llama_index.llms.ollama import Ollama

logging.basicConfig(level=logging.INFO)

if 'messages' not in st.session_state:
    st.session_state.messages = []

def stream_chat(model, messages):
    try:
        llm = Ollama(model=model, request_timeout=120.0)
        resp = llm.stream_chat(messages)
        response = ""
        response_placeholder = st.empty()

        for r in resp:
            response += r.delta
            response_placeholder.write(response)
        logging.info(f"Model: {model}, Messages: {messages}, Response: {response}")
        return response
    except Exception as e:
        logging.error(f"streaming error 발생: {str(e)}")
        raise e
    
def main():
    st.title("재령이가 만든 LLM 모델과 채팅하기 -3-")
    logging.info("app start")

    model = st.sidebar.selectbox("Select Model", ["llama3.2", "llama3", "llava", "gemma2"])
    logging.info("Selected model : {model}")

    # user history를 저장하기 위함
    if prompt := st.chat_input("질문해주세요!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        logging.info(f"user input: {prompt}")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                start_time = time.time()
                logging.info("응답 생성중")

        with st.spinner("응답 생성하는 중 ..."):
            try:
                messages = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in st.session_state.messages]
                response_message = stream_chat(model, messages)
                duration = time.time() - start_time
                respone_message_with_duration = f"{response_message}\n\nDuration: {duration:.2f} seconds"
                st.session_state.messages.append({"role": "assistant", "content": respone_message_with_duration})
                st.write(f"Duration: {duration:.2f} 초")
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": str(e)})
                logging.error(f"에러 발생 : {str(e)}")


if __name__ == "__main__":
    main()

# https://huggingface.co/spaces/upstage/open-ko-llm-leaderboard
# ↑ 한국어 모델이 필요한 경우