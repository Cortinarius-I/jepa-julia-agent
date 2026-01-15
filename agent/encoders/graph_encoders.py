"""
Graph encoders for Julia world state components.

This module provides specialized encoders for:
- Module dependency graphs
- Method dispatch graphs
- Type hierarchy graphs
- Call graphs

These encoders produce fixed-size embeddings that capture
structural properties of the codebase.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GINConv,
    GraphSAGE,
    global_mean_pool,
    global_max_pool,
)
from torch_geometric.data import Data, Batch


class ModuleGraphEncoder(nn.Module):
    """
    Encodes the module dependency graph.
    
    Each node represents a Julia module with features:
    - Name embedding
    - Export count
    - Import count
    - Is-submodule flag
    
    Edges represent dependencies (imports/uses).
    """
    
    def __init__(
        self,
        node_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.output_dim = output_dim
        
        # Node feature projection
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        
        # GAT layers for message passing
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * heads
            self.gat_layers.append(
                GATConv(
                    in_dim,
                    hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    concat=True if i < num_layers - 1 else False,
                )
            )
        
        # Final projection
        final_dim = hidden_dim * heads if num_layers > 1 else hidden_dim
        self.output_proj = nn.Linear(final_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode module graph to fixed-size embedding.
        
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment for multiple graphs [num_nodes]
            
        Returns:
            Graph embedding [batch_size, output_dim]
        """
        # Project node features
        h = self.node_proj(x)
        h = F.relu(h)
        
        # Apply GAT layers
        for i, gat in enumerate(self.gat_layers):
            h = gat(h, edge_index)
            if i < len(self.gat_layers) - 1:
                h = F.elu(h)
                h = self.dropout(h)
        
        # Global pooling
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        
        # Combine mean and max pooling
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_graph = (h_mean + h_max) / 2
        
        # Final projection
        output = self.output_proj(h_graph)
        
        return output


class DispatchGraphEncoder(nn.Module):
    """
    Encodes the method dispatch graph.
    
    Nodes represent methods, edges represent dispatch relationships:
    - Method A dispatches to method B for certain types
    - Ambiguity edges for overlapping signatures
    
    This captures Julia's multiple dispatch semantics.
    """
    
    def __init__(
        self,
        node_dim: int = 128,  # Method signature embedding
        edge_dim: int = 32,   # Edge type embedding
        hidden_dim: int = 256,
        output_dim: int = 512,
        num_layers: int = 4,
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim
        
        # Node projection
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        
        # Edge type embedding
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        
        # GIN layers (better at capturing structural patterns)
        self.gin_layers = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.gin_layers.append(GINConv(mlp, train_eps=True))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode dispatch graph.
        
        Args:
            x: Method node features [num_nodes, node_dim]
            edge_index: Dispatch edges [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim]
            batch: Batch assignment [num_nodes]
            
        Returns:
            Graph embedding [batch_size, output_dim]
        """
        # Project nodes
        h = self.node_proj(x)
        h = F.relu(h)
        
        # Apply GIN layers
        for gin in self.gin_layers:
            h = gin(h, edge_index)
            h = F.relu(h)
        
        # Global pooling
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        
        h_graph = global_mean_pool(h, batch)
        
        # Final projection
        output = self.output_proj(h_graph)
        
        return output


class TypeHierarchyEncoder(nn.Module):
    """
    Encodes the type hierarchy (subtype relationships).
    
    Nodes represent types, edges represent subtype relationships.
    Captures abstract vs concrete types, parametric types.
    """
    
    def __init__(
        self,
        node_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        
        # Use GraphSAGE for hierarchical data
        self.sage = GraphSAGE(
            in_channels=node_dim,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            out_channels=output_dim,
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode type hierarchy."""
        h = self.sage(x, edge_index)
        
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        
        return global_mean_pool(h, batch)


class CallGraphEncoder(nn.Module):
    """
    Encodes the static call graph.
    
    Nodes represent functions/methods, edges represent calls.
    Helps predict which methods might be affected by changes.
    """
    
    def __init__(
        self,
        node_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        
        self.encoder = ModuleGraphEncoder(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode call graph."""
        return self.encoder(x, edge_index, batch)


class MethodSignatureEncoder(nn.Module):
    """
    Encodes a single method signature to an embedding.
    
    Captures:
    - Function name
    - Argument types (ordered)
    - Return type hint (if available)
    - Where clauses (type constraints)
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,  # Type vocabulary
        embed_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 128,
        max_args: int = 10,
    ):
        super().__init__()
        
        self.max_args = max_args
        
        # Type embedding
        self.type_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Positional encoding for argument positions
        self.pos_embed = nn.Embedding(max_args + 2, embed_dim)  # +2 for name, return
        
        # Transformer for sequence
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=4,
                dim_feedforward=hidden_dim,
                batch_first=True,
            ),
            num_layers=2,
        )
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, output_dim)
        
    def forward(
        self,
        name_id: torch.Tensor,
        arg_type_ids: torch.Tensor,
        return_type_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode method signature.
        
        Args:
            name_id: Function name token ID [batch]
            arg_type_ids: Argument type IDs [batch, max_args]
            return_type_id: Return type ID [batch]
            
        Returns:
            Signature embedding [batch, output_dim]
        """
        batch_size = name_id.size(0)
        device = name_id.device
        
        # Build sequence: [name, arg1, arg2, ..., return]
        seq_len = 1 + self.max_args + (1 if return_type_id is not None else 0)
        
        # Embed types
        name_emb = self.type_embed(name_id).unsqueeze(1)  # [batch, 1, embed]
        arg_emb = self.type_embed(arg_type_ids)  # [batch, max_args, embed]
        
        if return_type_id is not None:
            return_emb = self.type_embed(return_type_id).unsqueeze(1)
            seq = torch.cat([name_emb, arg_emb, return_emb], dim=1)
        else:
            seq = torch.cat([name_emb, arg_emb], dim=1)
        
        # Add positional encoding
        positions = torch.arange(seq.size(1), device=device).unsqueeze(0)
        seq = seq + self.pos_embed(positions)
        
        # Transform
        h = self.transformer(seq)
        
        # Pool (take first token like BERT [CLS])
        h = h[:, 0, :]
        
        # Project
        output = self.output_proj(h)
        
        return output


class CompositeGraphEncoder(nn.Module):
    """
    Combines all graph encoders into a single world state encoder.
    
    This is the main encoder used by the JEPA model to encode
    complete world states into fixed-size embeddings.
    """
    
    def __init__(
        self,
        module_dim: int = 256,
        dispatch_dim: int = 512,
        type_dim: int = 128,
        call_dim: int = 256,
        output_dim: int = 512,
    ):
        super().__init__()
        
        self.module_encoder = ModuleGraphEncoder(output_dim=module_dim)
        self.dispatch_encoder = DispatchGraphEncoder(output_dim=dispatch_dim)
        self.type_encoder = TypeHierarchyEncoder(output_dim=type_dim)
        self.call_encoder = CallGraphEncoder(output_dim=call_dim)
        
        total_dim = module_dim + dispatch_dim + type_dim + call_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
        )
        
    def forward(
        self,
        module_graph: Data,
        dispatch_graph: Data,
        type_graph: Data,
        call_graph: Data,
    ) -> torch.Tensor:
        """
        Encode all graphs and fuse into single embedding.
        
        Returns:
            World state embedding [batch_size, output_dim]
        """
        # Encode each graph
        h_module = self.module_encoder(
            module_graph.x, module_graph.edge_index, 
            getattr(module_graph, 'batch', None)
        )
        
        h_dispatch = self.dispatch_encoder(
            dispatch_graph.x, dispatch_graph.edge_index,
            getattr(dispatch_graph, 'edge_attr', None),
            getattr(dispatch_graph, 'batch', None)
        )
        
        h_type = self.type_encoder(
            type_graph.x, type_graph.edge_index,
            getattr(type_graph, 'batch', None)
        )
        
        h_call = self.call_encoder(
            call_graph.x, call_graph.edge_index,
            getattr(call_graph, 'batch', None)
        )
        
        # Concatenate
        h_combined = torch.cat([h_module, h_dispatch, h_type, h_call], dim=-1)
        
        # Fuse
        output = self.fusion(h_combined)
        
        return output
