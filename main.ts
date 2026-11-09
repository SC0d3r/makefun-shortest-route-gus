/**
 * make.fun shortest-route-01 entry point.
 *
 * Pipeline: exact squared-distance nearest-neighbour construction, K=10
 * candidate lists, candidate-list 2-opt with a don't-look queue, Or-opt for
 * segments 1..3 in both orientations, and a deterministic iterated local
 * search with double-bridge perturbations.
 *
 * Determinism and bounds: all searches use fixed visit/round/ILS caps and a
 * mulberry32 PRNG seeded from a constant hash of the instance id and n. There
 * are no clocks, time limits, or unseeded random values. NN and candidate-list
 * construction are O(n^2); the candidate-list local search is O((n + moves) *
 * K) gain tests, with each accepted move costing O(min(arc length, n/2)) for
 * 2-opt or O(n) for an Or-opt splice. Memory is O(n*K) typed arrays.
 */

import { mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const CANDIDATE_COUNT = 10;
const TWO_OPT_PASSES = 80;
const OR_OPT_PASSES = 24;
const LOCAL_ROUNDS = 8;
const MAX_OR_OPT_SEGMENT = 3;

interface CandidateData {
  cities: Int32Array;
  squared: Float64Array;
  rounded: Float64Array;
  count: Int32Array;
}

interface Instance {
  id: string;
  n: number;
  x: Float64Array;
  y: Float64Array;
}

function readInstance(path: string, id: string): Instance {
  const tokens = readFileSync(path, "utf8").trim().split(/\s+/);
  if (tokens.length === 0 || tokens[0] === "") {
    throw new Error(`empty instance: ${path}`);
  }
  const n = Number(tokens[0]);
  if (!Number.isInteger(n) || n < 1 || tokens.length < 1 + 2 * n) {
    throw new Error(`invalid instance header or coordinates: ${path}`);
  }
  const x = new Float64Array(n);
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    x[i] = Number(tokens[1 + 2 * i]);
    y[i] = Number(tokens[2 + 2 * i]);
  }
  return { id, n, x, y };
}

function makeDistance(x: Float64Array, y: Float64Array): (a: number, b: number) => number {
  return (a: number, b: number): number => {
    const dx = x[a] - x[b];
    const dy = y[a] - y[b];
    return Math.round(Math.sqrt(dx * dx + dy * dy));
  };
}

function nearestNeighbourTour(n: number, x: Float64Array, y: Float64Array): Int32Array {
  const tour = new Int32Array(n);
  const used = new Uint8Array(n);
  if (n === 0) return tour;

  used[0] = 1;
  tour[0] = 0;
  for (let step = 1; step < n; step++) {
    let best = -1;
    let bestSquared = Number.POSITIVE_INFINITY;
    for (let city = 0; city < n; city++) {
      if (used[city] !== 0) continue;
      const dx = x[city] - x[tour[step - 1]];
      const dy = y[city] - y[tour[step - 1]];
      const squared = dx * dx + dy * dy;
      if (squared < bestSquared || (squared === bestSquared && city < best)) {
        bestSquared = squared;
        best = city;
      }
    }
    used[best] = 1;
    tour[step] = best;
  }
  return tour;
}

function buildCandidateLists(n: number, x: Float64Array, y: Float64Array): CandidateData {
  const width = Math.min(CANDIDATE_COUNT, Math.max(0, n - 1));
  const cities = new Int32Array(n * width);
  const squared = new Float64Array(n * width);
  const rounded = new Float64Array(n * width);
  const count = new Int32Array(n);

  const insert = (city: number, other: number, distanceSquared: number): void => {
    if (width === 0) return;
    const base = city * width;
    let length = count[city];
    if (length < width) {
      let at = base + length;
      while (at > base) {
        const previousSquared = squared[at - 1];
        const previousCity = cities[at - 1];
        if (
          distanceSquared > previousSquared ||
          (distanceSquared === previousSquared && other > previousCity)
        ) {
          break;
        }
        squared[at] = previousSquared;
        cities[at] = previousCity;
        at--;
      }
      squared[at] = distanceSquared;
      cities[at] = other;
      count[city] = length + 1;
      return;
    }

    const last = base + width - 1;
    const lastSquared = squared[last];
    const lastCity = cities[last];
    if (distanceSquared > lastSquared || (distanceSquared === lastSquared && other > lastCity)) {
      return;
    }
    let at = last;
    while (at > base) {
      const previousSquared = squared[at - 1];
      const previousCity = cities[at - 1];
      if (
        distanceSquared > previousSquared ||
        (distanceSquared === previousSquared && other > previousCity)
      ) {
        break;
      }
      squared[at] = previousSquared;
      cities[at] = previousCity;
      at--;
    }
    squared[at] = distanceSquared;
    cities[at] = other;
  };

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dx = x[i] - x[j];
      const dy = y[i] - y[j];
      const distanceSquared = dx * dx + dy * dy;
      insert(i, j, distanceSquared);
      insert(j, i, distanceSquared);
    }
  }

  for (let city = 0; city < n; city++) {
    const base = city * width;
    for (let k = 0; k < count[city]; k++) {
      rounded[base + k] = Math.round(Math.sqrt(squared[base + k]));
    }
  }

  return { cities, squared, rounded, count };
}

function tourLength(tour: Int32Array, n: number, distance: (a: number, b: number) => number): number {
  let total = 0;
  for (let i = 0; i < n; i++) {
    total += distance(tour[i], tour[(i + 1) % n]);
  }
  return total;
}

function reverseCircular(
  tour: Int32Array,
  position: Int32Array,
  n: number,
  start: number,
  length: number,
): void {
  const half = length >> 1;
  for (let k = 0; k < half; k++) {
    const left = (start + k) % n;
    const right = (start + length - 1 - k) % n;
    const a = tour[left];
    const b = tour[right];
    tour[left] = b;
    tour[right] = a;
    position[b] = left;
    position[a] = right;
  }
}

function twoOpt(
  tour: Int32Array,
  position: Int32Array,
  n: number,
  distance: (a: number, b: number) => number,
  candidates: CandidateData,
): number {
  if (n < 4) return 0;

  const width = Math.min(CANDIDATE_COUNT, n - 1);
  const queue = new Int32Array(n);
  const queued = new Uint8Array(n);
  let head = 0;
  let tail = 0;
  let queueSize = 0;

  const enqueue = (city: number): void => {
    if (queued[city] !== 0) return;
    queued[city] = 1;
    queue[tail] = city;
    tail = (tail + 1) % n;
    queueSize++;
  };

  for (let city = 0; city < n; city++) enqueue(city);

  let moves = 0;
  let visits = 0;
  const maxVisits = TWO_OPT_PASSES * n + 1024;

  while (queueSize > 0 && visits < maxVisits) {
    const a = queue[head];
    head = (head + 1) % n;
    queueSize--;
    queued[a] = 0;
    visits++;

    const pa = position[a];
    const pc = (pa + 1) % n;
    const c = tour[pc];
    const dac = distance(a, c);
    const base = a * width;

    for (let k = 0; k < candidates.count[a]; k++) {
      const b = candidates.cities[base + k];
      if (b === a || b === c) continue;

      const pb = position[b];
      const pd = (pb + 1) % n;
      const d = tour[pd];
      if (d === a) continue;

      const dab = candidates.rounded[base + k];
      const gain = dac + distance(b, d) - dab - distance(c, d);
      if (gain <= 0) continue;

      let firstArc = (pb - pc + n) % n;
      firstArc += 1;
      const secondArc = n - firstArc;
      if (firstArc <= secondArc) {
        reverseCircular(tour, position, n, pc, firstArc);
      } else {
        reverseCircular(tour, position, n, pd, secondArc);
      }

      enqueue(a);
      enqueue(c);
      enqueue(b);
      enqueue(d);
      moves++;
      break;
    }
  }

  if (process.env.TSP_DEBUG) console.error(`  2opt moves=${moves} visits=${visits}`);
  return moves;
}

function applyOrOpt(
  tour: Int32Array,
  position: Int32Array,
  n: number,
  segmentStart: number,
  segmentLength: number,
  insertAfter: number,
  reverseSegment: boolean,
  outside: Int32Array,
  rebuilt: Int32Array,
): void {
  const outsideStart = (segmentStart + segmentLength) % n;
  const outsideLength = n - segmentLength;
  for (let i = 0; i < outsideLength; i++) {
    outside[i] = tour[(outsideStart + i) % n];
  }
  const insertIndex = (insertAfter - outsideStart + n) % n;
  const segmentEnd = (segmentStart + segmentLength - 1) % n;

  for (let i = 0; i <= insertIndex; i++) {
    rebuilt[i] = outside[i];
  }
  for (let i = 0; i < segmentLength; i++) {
    const source = reverseSegment ? segmentEnd - i : segmentStart + i;
    rebuilt[insertIndex + 1 + i] = tour[(source + n) % n];
  }
  for (let i = insertIndex + 1; i < outsideLength; i++) {
    rebuilt[segmentLength + i] = outside[i];
  }
  tour.set(rebuilt);
  for (let i = 0; i < n; i++) position[tour[i]] = i;
}

function orOpt(
  tour: Int32Array,
  position: Int32Array,
  n: number,
  distance: (a: number, b: number) => number,
  candidates: CandidateData,
): number {
  if (n < 3) return 0;

  const width = Math.min(CANDIDATE_COUNT, n - 1);
  const queue = new Int32Array(n);
  const queued = new Uint8Array(n);
  const outside = new Int32Array(n);
  const rebuilt = new Int32Array(n);
  let debugLength = tourLength(tour, n, distance);
  let head = 0;
  let tail = 0;
  let queueSize = 0;

  const enqueue = (city: number): void => {
    if (queued[city] !== 0) return;
    queued[city] = 1;
    queue[tail] = city;
    tail = (tail + 1) % n;
    queueSize++;
  };

  for (let city = 0; city < n; city++) enqueue(city);

  let moves = 0;
  let visits = 0;
  const maxVisits = OR_OPT_PASSES * n + 1024;

  while (queueSize > 0 && visits < maxVisits) {
    const city = queue[head];
    head = (head + 1) % n;
    queueSize--;
    queued[city] = 0;
    visits++;

    const segmentStart = position[city];
    let accepted = false;

    for (
      let segmentLength = 1;
      segmentLength <= MAX_OR_OPT_SEGMENT && segmentLength <= n - 2;
      segmentLength++
    ) {
      const segmentEnd = (segmentStart + segmentLength - 1) % n;
      const p = tour[(segmentStart - 1 + n) % n];
      const q = tour[(segmentEnd + 1) % n];
      const first = tour[segmentStart];
      const last = tour[segmentEnd];
      const pCity = p;
      const qCity = q;
      const removeGain =
        distance(pCity, first) + distance(last, qCity) - distance(pCity, qCity);

      const inSegment = (pos: number): boolean => ((pos - segmentStart + n) % n) < segmentLength;

      for (let endpointIndex = 0; endpointIndex < (segmentLength === 1 ? 1 : 2); endpointIndex++) {
        const endpoint = endpointIndex === 0 ? first : last;
        const base = endpoint * width;
        for (let k = 0; k < candidates.count[endpoint]; k++) {
          const neighbour = candidates.cities[base + k];
          const neighbourPosition = position[neighbour];
          if (inSegment(neighbourPosition)) continue;

          const nextPosition = (neighbourPosition + 1) % n;
          const nextCity = tour[nextPosition];
          const previousPosition = (neighbourPosition - 1 + n) % n;
          const previousCity = tour[previousPosition];

          for (let side = 0; side < 2; side++) {
            const edgeStart = side === 0 ? neighbourPosition : previousPosition;
            const edgeEnd = side === 0 ? nextPosition : neighbourPosition;
            if (inSegment(edgeStart) || inSegment(edgeEnd)) continue;

            const edgeStartCity = tour[edgeStart];
            const edgeEndCity = tour[edgeEnd];
            if (
              edgeStartCity === pCity ||
              edgeEndCity === pCity ||
              edgeStartCity === qCity ||
              edgeEndCity === qCity
            ) {
              continue;
            }
            const edgeDistance = distance(edgeStartCity, edgeEndCity);

            for (let reverseSegment = 0; reverseSegment < 2; reverseSegment++) {
              const insertedFirst = reverseSegment === 0 ? first : last;
              const insertedLast = reverseSegment === 0 ? last : first;
              const insertCost =
                distance(edgeStartCity, insertedFirst) +
                distance(insertedLast, edgeEndCity) -
                edgeDistance;
              const delta = removeGain - insertCost;
              if (delta <= 0) continue;

              const beforeMoveLength = process.env.TSP_DEBUG
                ? tourLength(tour, n, distance)
                : 0;

              applyOrOpt(
                tour,
                position,
                n,
                segmentStart,
                segmentLength,
                edgeStart,
                reverseSegment !== 0,
                outside,
                rebuilt,
              );

              if (process.env.TSP_DEBUG) {
                const afterMoveLength = tourLength(tour, n, distance);
                if (afterMoveLength !== beforeMoveLength - delta) {
                  console.error(`debug tour: ${Array.from(tour).join(",")}`);
                  console.error(
                    `debug detail first=${first} last=${last} p=${pCity} q=${qCity} edgeStart=${edgeStartCity} edgeEnd=${edgeEndCity} positions seg=${segmentStart} cand=${neighbourPosition}`,
                  );
                  console.error(
                    `OROPT BAD delta=${delta} before=${beforeMoveLength} after=${afterMoveLength} segStart=${segmentStart} segLen=${segmentLength} insertAfter=${edgeStart} rev=${reverseSegment} tourLen=${debugLength}`,
                  );
                  throw new Error("Or-opt move mismatch");
                }
                debugLength = afterMoveLength;
              }

              enqueue(first);
              enqueue(last);
              enqueue(pCity);
              enqueue(qCity);
              enqueue(edgeStartCity);
              enqueue(edgeEndCity);
              moves++;
              accepted = true;
              break;
            }
            if (accepted) break;
          }
          if (accepted) break;
        }
        if (accepted) break;
      }
      if (accepted) break;
    }
  }

  if (process.env.TSP_DEBUG) console.error(`  oropt moves=${moves} visits=${visits}`);
  return moves;
}

function improve(
  tour: Int32Array,
  position: Int32Array,
  n: number,
  distance: (a: number, b: number) => number,
  candidates: CandidateData,
): number {
  let total = 0;
  for (let round = 0; round < LOCAL_ROUNDS; round++) {
    if (process.env.TSP_DEBUG) console.error(` round ${round}`);
    const twoOptMoves = twoOpt(tour, position, n, distance, candidates);
    if (process.env.TSP_DEBUG) {
      for (let i = 0; i < n; i++) {
        if (position[tour[i]] !== i) throw new Error(`position mismatch after 2opt at ${i}`);
      }
    }
    const orOptMoves = orOpt(tour, position, n, distance, candidates);
    if (process.env.TSP_DEBUG) {
      for (let i = 0; i < n; i++) {
        if (position[tour[i]] !== i) throw new Error(`position mismatch after oropt at ${i}`);
      }
    }
    total += twoOptMoves + orOptMoves;
    if (twoOptMoves === 0 && orOptMoves === 0) break;
  }
  return total;
}

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return (): number => {
    state = (state + 0x6d2b79f5) | 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function seedFor(id: string, n: number): number {
  let hash = 2166136261 >>> 0;
  for (let i = 0; i < id.length; i++) {
    hash = Math.imul(hash ^ id.charCodeAt(i), 16777619) >>> 0;
  }
  hash = Math.imul(hash ^ n, 16777619) >>> 0;
  return hash >>> 0;
}

function doubleBridge(
  tour: Int32Array,
  position: Int32Array,
  n: number,
  random: () => number,
  rebuilt: Int32Array,
): void {
  if (n < 4) return;

  const firstCut = Math.floor(random() * (n - 1));
  let secondCut = Math.floor(random() * (n - 2));
  if (secondCut >= firstCut) secondCut++;
  const low = Math.min(firstCut, secondCut);
  const high = Math.max(firstCut, secondCut);
  let thirdCut = Math.floor(random() * (n - 3));
  if (thirdCut >= low) thirdCut++;
  if (thirdCut >= high) thirdCut++;

  const cuts = [firstCut, secondCut, thirdCut].sort((a, b) => a - b);
  const c1 = cuts[0];
  const c2 = cuts[1];
  const c3 = cuts[2];

  let at = 0;
  for (let i = 0; i <= c1; i++) rebuilt[at++] = tour[i];
  for (let i = c2 + 1; i <= c3; i++) rebuilt[at++] = tour[i];
  for (let i = c1 + 1; i <= c2; i++) rebuilt[at++] = tour[i];
  for (let i = c3 + 1; i < n; i++) rebuilt[at++] = tour[i];

  tour.set(rebuilt);
  for (let i = 0; i < n; i++) position[tour[i]] = i;
}

function solve(instance: Instance): Int32Array {
  const { n, x, y } = instance;
  const distance = makeDistance(x, y);
  const tour = nearestNeighbourTour(n, x, y);
  if (n <= 1) return tour;

  const position = new Int32Array(n);
  for (let i = 0; i < n; i++) position[tour[i]] = i;

  const candidates = buildCandidateLists(n, x, y);
  if (process.env.TSP_DEBUG) console.error(`candidates done n=${n}`);
  improve(tour, position, n, distance, candidates);
  if (process.env.TSP_DEBUG) console.error(`local done len=${tourLength(tour, n, distance)}`);

  let best = tour.slice();
  let bestLength = tourLength(best, n, distance);
  const rounds = n <= 3000 ? 4 : n <= 5000 ? 2 : 1;
  const random = mulberry32(seedFor(instance.id, n));
  const work = new Int32Array(n);
  const workPosition = new Int32Array(n);
  const rebuilt = new Int32Array(n);

  for (let round = 0; round < rounds; round++) {
    if (process.env.TSP_DEBUG) console.error(`ils ${round} best=${bestLength}`);
    work.set(best);
    for (let i = 0; i < n; i++) workPosition[work[i]] = i;
    doubleBridge(work, workPosition, n, random, rebuilt);
    improve(work, workPosition, n, distance, candidates);
    const length = tourLength(work, n, distance);
    if (process.env.TSP_DEBUG) console.error(`ils ${round} result=${length}`);
    if (length < bestLength) {
      bestLength = length;
      best.set(work);
    }
  }

  return best;
}

function main(): void {
  const inputDirectory = process.env.MAKEFUN_INPUT_DIR ?? "/input";
  const outputDirectory = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";
  mkdirSync(outputDirectory, { recursive: true });

  const inputNames = readdirSync(inputDirectory)
    .filter((name) => name.endsWith(".txt"))
    .sort();

  for (const name of inputNames) {
    const id = name.slice(0, -4);
    const instance = readInstance(join(inputDirectory, name), id);
    const tour = solve(instance);
    writeFileSync(join(outputDirectory, `${id}.tour`), `${Array.from(tour).join("\n")}\n`);
  }
}

main();
