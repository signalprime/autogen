"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[8866],{1831:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>c,contentTitle:()=>o,default:()=>h,frontMatter:()=>r,metadata:()=>a,toc:()=>l});var s=t(5893),i=t(1151);const r={sidebar_label:"web_archiver_agent",title:"agentchat.contrib.web_archiver_agent"},o=void 0,a={id:"reference/agentchat/contrib/web_archiver_agent",title:"agentchat.contrib.web_archiver_agent",description:"WebArchiverAgent",source:"@site/docs/reference/agentchat/contrib/web_archiver_agent.md",sourceDirName:"reference/agentchat/contrib",slug:"/reference/agentchat/contrib/web_archiver_agent",permalink:"/autogen/docs/reference/agentchat/contrib/web_archiver_agent",draft:!1,unlisted:!1,editUrl:"https://github.com/microsoft/autogen/edit/main/website/docs/reference/agentchat/contrib/web_archiver_agent.md",tags:[],version:"current",frontMatter:{sidebar_label:"web_archiver_agent",title:"agentchat.contrib.web_archiver_agent"},sidebar:"referenceSideBar",previous:{title:"text_analyzer_agent",permalink:"/autogen/docs/reference/agentchat/contrib/text_analyzer_agent"},next:{title:"web_surfer",permalink:"/autogen/docs/reference/agentchat/contrib/web_surfer"}},c={},l=[{value:"WebArchiverAgent",id:"webarchiveragent",level:2},{value:"__init__",id:"__init__",level:3},{value:"classifier_to_collector_reply",id:"classifier_to_collector_reply",level:3},{value:"collect_content",id:"collect_content",level:3}];function d(e){const n={code:"code",em:"em",h2:"h2",h3:"h3",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,i.a)(),...e.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(n.h2,{id:"webarchiveragent",children:"WebArchiverAgent"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"class WebArchiverAgent(ConversableAgent)\n"})}),"\n",(0,s.jsx)(n.h3,{id:"__init__",children:"__init__"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:'def __init__(silent: bool = True,\n             storage_path: str = "./content",\n             max_depth: int = 1,\n             page_load_time: float = 6,\n             *args,\n             **kwargs)\n'})}),"\n",(0,s.jsx)(n.p,{children:"WebArchiverAgent: Custom LLM agent for collecting online content."}),"\n",(0,s.jsx)(n.p,{children:"The WebArchiverAgent class is a custom Autogen agent that can be used to collect and store online content from different\nweb pages. It extends the ConversableAgent class and provides additional functionality for managing a list of\nadditional links, storing collected content in local directories, and customizing request headers.  WebArchiverAgent\nuses deque to manage a list of additional links for further exploration, with a maximum depth limit set by max_depth\nparameter. The collected content is stored in the specified storage path (storage_path) using local directories.\nWebArchiverAgent can be customized with request_kwargs and llm_config parameters during instantiation. The default\nUser-Agent header is used for requests, but it can be overridden by providing a new dictionary of headers under\nrequest_kwargs."}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"silent"})," ",(0,s.jsx)(n.em,{children:"bool"})," - If True, the agent operates in silent mode with minimal output. Defaults to True."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"storage_path"})," ",(0,s.jsx)(n.em,{children:"str"})," - The path where the collected content will be stored. Defaults to './content'."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"max_depth"})," ",(0,s.jsx)(n.em,{children:"int"})," - Maximum depth limit for exploring additional links from a web page. This defines how deep\nthe agent will go into linked pages from the starting point. Defaults to 1."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"page_load_time"})," ",(0,s.jsx)(n.em,{children:"float"})," - Time in seconds to wait for loading each web page. This ensures that dynamic content\nhas time to load before the page is processed. Defaults to 6 seconds.\n*args, **kwargs: Additional arguments and keyword arguments to be passed to the parent class ",(0,s.jsx)(n.code,{children:"ConversableAgent"}),".\nThese can be used to configure underlying behaviors of the agent that are not explicitly\ncovered by the constructor's parameters."]}),"\n"]}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Notes"}),":"]}),"\n",(0,s.jsxs)(n.p,{children:["The ",(0,s.jsx)(n.code,{children:"silent"})," parameter can be useful for controlling the verbosity of the agent's operations, particularly\nin environments where logging or output needs to be minimized for performance or clarity."]}),"\n",(0,s.jsx)(n.p,{children:"Software Dependencies:"}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsx)(n.li,{children:"requests"}),"\n",(0,s.jsx)(n.li,{children:"beautifulsoup4"}),"\n",(0,s.jsx)(n.li,{children:"pdfminer"}),"\n",(0,s.jsx)(n.li,{children:"selenium"}),"\n",(0,s.jsx)(n.li,{children:"arxiv"}),"\n",(0,s.jsx)(n.li,{children:"pillow"}),"\n"]}),"\n",(0,s.jsx)(n.h3,{id:"classifier_to_collector_reply",children:"classifier_to_collector_reply"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"def classifier_to_collector_reply(recipient: Agent, messages: Union[List[str],\n                                                                    str],\n                                  sender: Agent,\n                                  config: dict) -> Tuple[bool, str]\n"})}),"\n",(0,s.jsx)(n.p,{children:"Processes the last message in a conversation to generate a boolean classification response."}),"\n",(0,s.jsx)(n.p,{children:'This method takes the most recent message from a conversation, uses the recipient\'s method to generate a reply,\nand classifies the reply as either "True" or "False" based on its content. It is designed for scenarios where\nthe reply is expected to represent a boolean value, simplifying downstream processing.'}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"recipient"})," ",(0,s.jsx)(n.em,{children:"Agent"})," - The agent or object responsible for generating replies. Must have a method ",(0,s.jsx)(n.code,{children:"generate_oai_reply"}),"\nthat accepts a list of messages, a sender, and optionally a configuration, and returns a tuple\nwhere the second element is the reply string."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"messages"})," ",(0,s.jsx)(n.em,{children:"Union[List[str], str]"})," - A list of messages or a single message string from the conversation. The last message\nin this list is used to generate the reply."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"sender"})," ",(0,s.jsx)(n.em,{children:"Agent"})," - The entity that sent the message. This could be an identifier, an object, or any representation\nthat the recipient's reply generation method expects."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"config"})," ",(0,s.jsx)(n.em,{children:"dict"})," - Configuration parameters for the reply generation process, if required by the recipient's method."]}),"\n"]}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,s.jsx)(n.p,{children:'Tuple[bool, str]: A tuple containing a boolean status (always True in this implementation) and the classification result\nas "True" or "False" based on the content of the generated reply.'}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Notes"}),":"]}),"\n",(0,s.jsx)(n.p,{children:'The classification is case-insensitive and defaults to "False" if the reply does not explicitly contain\n"true" or "false". This behavior ensures a conservative approach to classification.'}),"\n",(0,s.jsx)(n.h3,{id:"collect_content",children:"collect_content"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"def collect_content(recipient: Agent, messages: Union[List[str], str],\n                    sender: Agent, config: dict) -> Tuple[bool, str]\n"})}),"\n",(0,s.jsx)(n.p,{children:"Collects and archives content from links found in messages."}),"\n",(0,s.jsx)(n.p,{children:"This function scans messages for URLs, fetches content from these URLs,\nand archives them to a specified local directory. It supports recursive\nlink fetching up to a defined depth."}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsx)(n.li,{children:"recipient (Agent): The agent designated to receive the content."}),"\n",(0,s.jsx)(n.li,{children:"messages (Union[List[str], str]): A list of messages or a single message containing URLs."}),"\n",(0,s.jsx)(n.li,{children:"sender (Agent): The agent sending the content."}),"\n",(0,s.jsx)(n.li,{children:"config (dict): Configuration parameters for content fetching and archiving."}),"\n"]}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsx)(n.li,{children:"Tuple[bool, str]: A tuple where the first element is a boolean indicating\nsuccess or failure, and the second element is a string message detailing\nthe outcome or providing error logs in case of failure."}),"\n"]})]})}function h(e={}){const{wrapper:n}={...(0,i.a)(),...e.components};return n?(0,s.jsx)(n,{...e,children:(0,s.jsx)(d,{...e})}):d(e)}},1151:(e,n,t)=>{t.d(n,{Z:()=>a,a:()=>o});var s=t(7294);const i={},r=s.createContext(i);function o(e){const n=s.useContext(r);return s.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function a(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(i):e.components||i:o(e.components),s.createElement(r.Provider,{value:n},e.children)}}}]);