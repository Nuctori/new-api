/*
Copyright (C) 2025 QuantumNous

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

For commercial licensing, please contact support@quantumnous.com
*/
import assert from 'node:assert/strict';
import { describe, test } from 'node:test';
import {
  readResponsesCompatMode,
  writeResponsesCompatMode,
} from './channel-settings';

describe('classic channel responses compatibility helpers', () => {
  test('defaults to disabled when settings are absent or invalid', () => {
    assert.equal(readResponsesCompatMode(''), false);
    assert.equal(readResponsesCompatMode('{invalid'), false);
  });

  test('reads explicit enabled state from settings JSON', () => {
    assert.equal(
      readResponsesCompatMode('{"responses_compat_mode":true}'),
      true,
    );
  });

  test('writes explicit false for OpenAI-compatible channel types', () => {
    const settings = writeResponsesCompatMode({ keep: 'value' }, 43, false);
    assert.deepEqual(settings, {
      keep: 'value',
      responses_compat_mode: false,
    });
  });

  test('removes the field for unsupported channel types', () => {
    const settings = writeResponsesCompatMode(
      { responses_compat_mode: true, keep: 'value' },
      14,
      true,
    );
    assert.deepEqual(settings, { keep: 'value' });
  });
});
