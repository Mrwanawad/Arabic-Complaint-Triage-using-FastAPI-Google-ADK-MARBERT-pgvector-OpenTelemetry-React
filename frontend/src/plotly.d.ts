declare module "plotly.js-basic-dist-min" {
  const Plotly: {
    react(node: HTMLElement, data: unknown[], layout: unknown, config: unknown): Promise<unknown>;
    purge(node: HTMLElement): void;
  };
  export default Plotly;
}
