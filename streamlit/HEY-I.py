import streamlit as st

st.set_page_config(page_icon="🔎", layout="wide")
st.header("안녕하세요😁 당신만의 면접 도우미, **HEY-I**_v2.0 입니다!")

st.subheader("이 사이트는 inference 결과를 미리 볼 수 있는 사이트입니다.")

with st.expander("실제 페이지에서는 다음과 같은 과정으로 로그인을 할 수 있습니다."):
    with st.form("my_form"):
        name = st.text_input("이름을 입력하세요")
        num = st.number_input("원하는 네 자리 수를 입력하세요", min_value=1000, max_value=9999)

        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted:
            if len(name) > 1:
                st.session_state.name = name
                st.session_state.num = num
                st.success(f"{name}_{num} 확인됐습니다! 다음 단계로 넘어가세요!", icon="👍")
                print(st.session_state.name, st.session_state.num)
            else:
                st.warning("이름을 두 글자 이상 입력해주세요!", icon="⚠️")

        else:
            st.warning("입력해주세요!", icon="⚠️")
