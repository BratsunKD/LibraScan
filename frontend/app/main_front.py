import os
import gradio as gr
import requests

classes = ["обложка", "титульный лиcт", "с аннотациями", "страница с выпускными данными"]

predict_url = os.getenv("PREDICT_URL", "http://localhost:8000/predict")
save_answer_url = os.getenv("SAVE_ANSWER_URL", "http://localhost:8000/save_answer")


def classify_image(image):
    if image is None:
        return None, gr.update(visible=False), gr.update(visible=False), None, gr.update(visible=False), gr.update(visible=False)

    with open(image, 'rb') as f:
        files = {'file': f}
        try:
            response = requests.post(predict_url, files=files)
            response.raise_for_status()
            result = response.json()

            predicted_index = result.get('class')
            prediction_hash = result.get('hash', None)

            # Приводим индекс к имени, если он валиден
            if isinstance(predicted_index, int) and 0 <= predicted_index < len(classes):
                predicted_label = classes[predicted_index]
                selected_value = predicted_label
            else:
                predicted_label = "Не определено"
                selected_value = None

            return (
                predicted_label,  # отображаем имя предсказанного класса
                gr.update(visible=True, choices=classes, value=selected_value),  # отображаем список имён
                gr.update(visible=True),
                prediction_hash,
                gr.update(visible=False),
                gr.update(visible=True)
            )
        except Exception as e:
            print("Ошибка при отправке:", e)
            return (
                "Ошибка при анализе",
                gr.update(visible=False),
                gr.update(visible=False),
                None,
                gr.update(visible=False),
                gr.update(visible=False)
            )


def submit_feedback(selected_class_name, prediction_hash):
    if selected_class_name is None or prediction_hash is None:
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            None,
            gr.update(visible=False),
            gr.update(visible=True)
        )

    try:
        # Конвертируем имя в индекс
        user_answer_index = str(classes.index(selected_class_name))
        data = {
            "prediction_hash": prediction_hash,
            "user_answer": user_answer_index
        }
        response = requests.post(save_answer_url, data=data)
        response.raise_for_status()
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            None,
            gr.update(visible=True),
            gr.update(visible=False)
        )
    except Exception as e:
        print("Ошибка при отправке обратной связи:", e)
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True),
            prediction_hash,
            gr.update(visible=False),
            gr.update(visible=True)
        )


def reset_app():
    return (
        None,
        "",
        gr.update(visible=False),
        gr.update(visible=False),
        None,
        gr.update(visible=False),
        gr.update(visible=False)
    )


with gr.Blocks() as demo:
    gr.Markdown("## 🖼️ Классификатор изображений")

    image = gr.Image(type="filepath", label="Загрузите изображение")
    analyze_btn = gr.Button("Анализировать изображение")

    predicted_output = gr.Textbox(label="Предсказанный класс", interactive=False)
    manual_label = gr.Radio(choices=classes, label="Выберите правильный класс", visible=False)
    submit_feedback_btn = gr.Button("Подтвердить выбор", visible=False)
    feedback_message = gr.Markdown("✅ Спасибо за обратную связь!", visible=False)
    new_session_btn = gr.Button("Загрузить новое изображение", visible=False)
    load_another_btn = gr.Button("Загрузить ещё", visible=False)

    prediction_hash = gr.State()

    analyze_btn.click(
        classify_image,
        inputs=[image],
        outputs=[predicted_output, manual_label, submit_feedback_btn, prediction_hash, feedback_message, load_another_btn]
    )
    submit_feedback_btn.click(
        submit_feedback,
        inputs=[manual_label, prediction_hash],
        outputs=[manual_label, submit_feedback_btn, feedback_message, prediction_hash, new_session_btn, load_another_btn]
    )
    new_session_btn.click(
        reset_app,
        outputs=[image, predicted_output, manual_label, submit_feedback_btn, prediction_hash, new_session_btn, load_another_btn]
    )
    load_another_btn.click(
        reset_app,
        outputs=[image, predicted_output, manual_label, submit_feedback_btn, prediction_hash, new_session_btn, load_another_btn]
    )

demo.launch(
    server_name="0.0.0.0",  # Разрешаем доступ извне контейнера
    server_port=7860,       # Явно указываем порт
    share=False             # Отключаем публичный URL (если не нужен)
)

