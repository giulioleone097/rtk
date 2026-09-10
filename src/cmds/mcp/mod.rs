//! `tokenaut mcp`: an MCP server over stdio exposing the context tools.

mod exec;
mod fetch;
mod store;
mod tools;

use anyhow::{Context, Result};
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock, Implementation, ServerCapabilities, ServerInfo};
use rmcp::transport::stdio;
use rmcp::{tool, tool_handler, tool_router, ErrorData, ServerHandler, ServiceExt};

use exec::{ExecuteFileInput, ExecuteInput};
use fetch::FetchInput;
use tools::{BatchExecuteInput, SearchInput};

/// Stateless handler: every tool call opens its own index connection.
#[derive(Debug, Clone, Default)]
pub struct ContextServer;

#[tool_router]
impl ContextServer {
    /// Run shell commands in parallel, index their output, and answer the queries against the index.
    #[tool]
    async fn ctx_batch_execute(
        &self,
        Parameters(input): Parameters<BatchExecuteInput>,
    ) -> Result<CallToolResult, ErrorData> {
        text_result(tools::batch_execute(input).await)
    }

    /// Run shell, javascript or python code and return its output; large output is indexed instead.
    #[tool]
    async fn ctx_execute(
        &self,
        Parameters(input): Parameters<ExecuteInput>,
    ) -> Result<CallToolResult, ErrorData> {
        text_result(exec::execute(input).await)
    }

    /// Run code over one file, with FILE_PATH and FILE_CONTENT already set, and return its output.
    #[tool]
    async fn ctx_execute_file(
        &self,
        Parameters(input): Parameters<ExecuteFileInput>,
    ) -> Result<CallToolResult, ErrorData> {
        text_result(exec::execute_file(input).await)
    }

    /// Fetch URLs with curl, convert HTML or JSON to text, index them under fetch:<label> and return a preview.
    #[tool]
    async fn ctx_fetch_and_index(
        &self,
        Parameters(input): Parameters<FetchInput>,
    ) -> Result<CallToolResult, ErrorData> {
        // Blocking: curl and the index run off the reactor so a slow host does
        // not stall the server's other calls.
        let fetched = tokio::task::spawn_blocking(move || fetch::fetch_and_index(input))
            .await
            .map_err(|err| ErrorData::internal_error(format!("fetch task: {err}"), None))?;
        text_result(fetched)
    }

    /// Search everything indexed so far, one section list per query; `source` restricts to one label.
    #[tool]
    async fn ctx_search(
        &self,
        Parameters(input): Parameters<SearchInput>,
    ) -> Result<CallToolResult, ErrorData> {
        text_result(tools::search(input))
    }
}

#[tool_handler]
impl ServerHandler for ContextServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("tokenaut", env!("CARGO_PKG_VERSION")))
            .with_instructions(
                "ctx_batch_execute runs shell commands in parallel, indexes their output \
                 and answers queries against that index; ctx_execute and ctx_execute_file \
                 run one shell, javascript or python script and return its output, indexing \
                 it instead when it is large; ctx_fetch_and_index indexes web pages under \
                 fetch:<label>; ctx_search answers queries against everything indexed so far, \
                 optionally within one source label. Content already shown earlier in a response is replaced by \
                 a back-reference to where it first appeared.",
            )
    }
}

fn text_result(body: Result<String>) -> Result<CallToolResult, ErrorData> {
    match body {
        Ok(text) => Ok(CallToolResult::success(vec![ContentBlock::text(text)])),
        Err(err) => Err(ErrorData::internal_error(format!("{err:#}"), None)),
    }
}

/// Serve MCP over stdio until the client disconnects.
pub fn run() -> Result<i32> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let service = ContextServer
            .serve(stdio())
            .await
            .context("mcp: stdio transport")?;
        service.waiting().await.context("mcp: server loop")?;
        Ok(0)
    })
}
