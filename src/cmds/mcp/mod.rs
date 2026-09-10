//! `rtk mcp`: an MCP server over stdio exposing the context tools.

mod store;
mod tools;

use anyhow::{Context, Result};
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock};
use rmcp::transport::stdio;
use rmcp::{ErrorData, ServiceExt, tool, tool_router};

use tools::{BatchExecuteInput, SearchInput};

/// Stateless handler: every tool call opens its own index connection.
#[derive(Debug, Clone, Default)]
pub struct ContextServer;

#[tool_router(server_handler)]
impl ContextServer {
    /// Run shell commands in parallel, index their output, and answer the queries against the index.
    #[tool]
    async fn ctx_batch_execute(
        &self,
        Parameters(input): Parameters<BatchExecuteInput>,
    ) -> Result<CallToolResult, ErrorData> {
        text_result(tools::batch_execute(input).await)
    }

    /// Search the indexed command output, one section list per query.
    #[tool]
    async fn ctx_search(
        &self,
        Parameters(input): Parameters<SearchInput>,
    ) -> Result<CallToolResult, ErrorData> {
        text_result(tools::search(input))
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
