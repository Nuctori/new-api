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

export function readResponsesCompatMode(settingsText) {
  if (!settingsText) return false;
  try {
    const parsed = JSON.parse(settingsText);
    return parsed?.responses_compat_mode === true;
  } catch {
    return false;
  }
}

export function writeResponsesCompatMode(settings, channelType, enabled) {
  if (channelType === 1 || channelType === 43) {
    settings.responses_compat_mode = enabled === true;
  } else if ('responses_compat_mode' in settings) {
    delete settings.responses_compat_mode;
  }
  return settings;
}
