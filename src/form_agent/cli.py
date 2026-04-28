"""Click CLI: form-agent ingest|list|ask|summarize|ask-all."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from .agent import FormAgent

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@click.group()
@click.pass_context
def main(ctx: click.Context) -> None:
    """Intelligent Form Agent CLI."""
    ctx.ensure_object(dict)
    ctx.obj["agent"] = FormAgent()


@main.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.pass_context
def ingest(ctx: click.Context, paths: tuple[str, ...]) -> None:
    """Ingest one or more form files (PDF/TXT/image)."""
    agent: FormAgent = ctx.obj["agent"]
    for path in paths:
        click.echo(f"Ingesting {path} ...")
        form = agent.ingest(Path(path))
        click.secho(f"  -> id={form.id}  type={form.form_type}  fields={len(form.fields)}", fg="green")


@main.command("list")
@click.pass_context
def list_cmd(ctx: click.Context) -> None:
    """List ingested forms."""
    agent: FormAgent = ctx.obj["agent"]
    forms = agent.list_forms()
    if not forms:
        click.echo("(no forms ingested)")
        return
    for f in forms:
        click.echo(f"{f.id:40s}  {f.form_type:24s}  {Path(f.source_path).name}")


@main.command()
@click.argument("form_id")
@click.argument("question", nargs=-1, required=True)
@click.pass_context
def ask(ctx: click.Context, form_id: str, question: tuple[str, ...]) -> None:
    """Ask a question of a single form."""
    agent: FormAgent = ctx.obj["agent"]
    q = " ".join(question)
    ans = agent.ask(form_id, q)
    click.secho(f"\nQ: {q}", bold=True)
    click.secho(f"A: {ans.answer}", fg="cyan")
    click.echo(f"confidence: {ans.confidence:.2f}")
    if ans.citations:
        click.echo("citations:")
        for c in ans.citations:
            click.echo(f"  - field={c.field} page={c.page} snippet={c.snippet!r}")


@main.command()
@click.argument("form_id")
@click.pass_context
def summarize(ctx: click.Context, form_id: str) -> None:
    """Summarize a single form."""
    agent: FormAgent = ctx.obj["agent"]
    s = agent.summarize(form_id)
    click.secho("TL;DR:", bold=True)
    for b in s.tldr:
        click.echo(f"  - {b}")
    _section("Key parties", s.key_parties)
    _section("Key dates", s.key_dates)
    _section("Key amounts", s.key_amounts)
    _section("Obligations / actions", s.obligations_or_actions)
    _section("Risks / anomalies", s.risks_or_anomalies)
    if s.overall:
        click.secho("\nOverall:", bold=True)
        click.echo(s.overall)


@main.command("ask-all")
@click.argument("question", nargs=-1, required=True)
@click.option("-k", "--top-k", default=6, show_default=True)
@click.pass_context
def ask_all(ctx: click.Context, question: tuple[str, ...], top_k: int) -> None:
    """Ask a question across all ingested forms."""
    agent: FormAgent = ctx.obj["agent"]
    q = " ".join(question)
    ans = agent.ask_all(q, top_k=top_k)
    click.secho(f"\nQ: {q}", bold=True)
    click.secho(f"strategy: {ans.strategy}", fg="yellow")
    click.secho(f"A: {ans.answer}", fg="cyan")
    click.echo(f"confidence: {ans.confidence:.2f}")
    click.echo(f"forms considered: {len(ans.forms_considered)}")
    if ans.citations:
        click.echo("citations:")
        for c in ans.citations:
            click.echo(f"  - field={c.field} page={c.page} snippet={c.snippet!r}")


@main.command()
@click.argument("form_id")
@click.pass_context
def show(ctx: click.Context, form_id: str) -> None:
    """Print the stored JSON for a form."""
    agent: FormAgent = ctx.obj["agent"]
    form = agent.get_form(form_id)
    if form is None:
        raise click.ClickException(f"Unknown form id: {form_id}")
    click.echo(json.dumps(
        {"id": form.id, "form_type": form.form_type, "source_path": form.source_path,
         "fields": form.fields},
        indent=2, default=str,
    ))


def _section(title: str, items: list[str]) -> None:
    if not items:
        return
    click.secho(f"\n{title}:", bold=True)
    for it in items:
        click.echo(f"  - {it}")


if __name__ == "__main__":
    main(obj={})
