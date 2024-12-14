import os
import gradio as gr
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
import logging
logging.basicConfig(level=logging.INFO)

class RoadsideAssistanceChatbot:
    def __init__(self):
        load_dotenv()
        
        # Azure OpenAI Client
        self.openai_client = AzureOpenAI(
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            api_version="2023-12-01-preview"
        )
        
        # Azure Search Client
        credential = AzureKeyCredential(os.environ['AZURE_THL_SEARCH_KEY'])
        self.search_client = SearchClient(endpoint=os.environ['AZURE_THL_SEARCH_ENDPOINT'], index_name=os.environ['AZURE_THL_SEARCH_INDEX_NAME'], credential=credential)
    
    def retrieve_context(self, query, top_k=3):
        """
        Retrieve relevant context from Azure AI Search
        """
        search_results = self.search_client.search(
            search_text=query,
            top=top_k
        )
        context = "\n".join([result['content'] for result in search_results])
        
        return context
    

    def generate_roadside_assistance_prompt(self, context, query, country=None):
        """
        Generate a prompt for roadside assistance that considers country-specific information.
        
        Args:
            context (str): Background information about roadside assistance
            query (str): Specific roadside assistance query
            country (str, optional): Country of the campervan location
        
        Returns:
            str: A refined prompt tailored to country-specific roadside assistance
        """
        # If country is not provided, ask for it
        if not country:
            return f"""
            Context: {context}
            
            Road Side Care Assistance Query: {query}
            
            I notice that no specific country was mentioned. Could you please confirm:
            1. In which country is the campervan located?
            2. What specific make and model of the campervan (e.g., New Zealand Britz Hitop)?
            
            Once you provide these details, I can give you precise roadside assistance information.
            """
        
        # Country-specific prompt
        return f"""
        Context: {context}
        
        Country of Campervan: {country}
        Road Side Care Assistance Query: {query}
        
        Specific Instructions:
        - Focus ONLY on roadside assistance information for campervans in {country} based on the context
        - If multiple answers exist, list them including troubleshooting guide
        - Prioritize Tourism Holding Limited owned campervan and exclude other companies
        - If no specific information is available for {country}, clearly state: 
        "I apologize, but I do not have roadside assistance details for campervans in {country}."
        
        Response Requirements:
        - Provide your answer based on the country 
        - Provide concise, actionable information
        - Include emergency contact numbers if available
        """

    def get_roadside_assistance(self, context, query, country=None):
        """
        Retrieve roadside assistance information with country-specific filtering.
        
        Args:
            context (str): Background information
            query (str): Specific assistance query
            country (str, optional): Country of campervan location
        
        Returns:
            str: Roadside assistance response
        """
        prompt = self.generate_roadside_assistance_prompt(context, query, country)
        logging.info(prompt)

        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a specialized roadside assistance AI for campervans, focusing on country-specific support."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        
        return response.choices[0].message.content

    def generate_response(self, query, history, country=None):
        """
        Generate RAG-powered response
        
        Args:
            query (str): User's query
            history (list): Conversation history
            country (str, optional): Country of campervan location
        
        Returns:
            tuple: Updated conversation history and cleared input
        """
        context = self.retrieve_context(query)

        ai_response = self.get_roadside_assistance(context, query, country)
        # Update conversation history
        updated_history = history + [[query, ai_response]]
        
        return updated_history, ""

    def create_interface(self):
        """
        Create Gradio Blocks Interface
        """
        with gr.Blocks(title="🚗 Roadside Care AI Assistant", theme="soft") as demo:
            gr.Markdown("""
                # THL Roadside Care AI Assistant
                ## Smart Roadside Support at Your Fingertips 🛠️
            """)

            # Chatbox to store conversation history
            chatbox = gr.Chatbot(
                value=[["system", "How can I help?"]],  # Start with empty history
                label="Conversation History",
                bubble_full_width=False,
                height=400
            )

            # Country dropdown
            country_dropdown = gr.Dropdown(
                label="Country", 
                choices=[
                    "New Zealand", 
                    "Australia", 
                    "United States", 
                    "Canada", 
                    "United Kingdom"
                ],
                value=None,
                allow_custom_value=True
            )
            
            # Input textbox
            query_input = gr.Textbox(label="Ask a Roadside Assistance Question")
            
            
            
            # Button row
            with gr.Row():
                submit_btn = gr.Button("Submit")
                clear_btn = gr.Button("Clear")
            
            # Submit button event
            submit_btn.click(
                fn=self.generate_response, 
                inputs=[query_input, chatbox, country_dropdown], 
                outputs=[chatbox, query_input]
            )
            
            # Clear button event 
            clear_btn.click(
                fn=lambda: ([], ""),  # Return empty history and empty query
                inputs=None, 
                outputs=[chatbox, query_input]
            )
            
            # Custom CSS for styling
            demo.css = """
            .gradio-container .chatbot .message {
                color: blue !important;
            }
            """
        
        return demo

def main():
    chatbot = RoadsideAssistanceChatbot()
    interface = chatbot.create_interface()
    interface.launch(share=True)

if __name__ == "__main__":
    main()