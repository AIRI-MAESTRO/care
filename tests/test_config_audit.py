"""Tests for CareConfig.audit_fields — flag misconfigured URL/model slots."""

from __future__ import annotations

from care.config import CareConfig, MageConfig, PlatformConfig

_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NSJ9.s3cr3tsig"


class TestAuditFields:
    def test_clean_config_has_no_warnings(self):
        assert CareConfig().audit_fields() == []

    def test_jwt_in_validation_model_flagged(self):
        cfg = CareConfig(platform=PlatformConfig(validation_model=_JWT))
        warnings = cfg.audit_fields()
        assert any("validation_model" in w and "JWT" in w for w in warnings)

    def test_jwt_in_base_url_flagged(self):
        cfg = CareConfig(platform=PlatformConfig(validation_base_url=_JWT))
        warnings = cfg.audit_fields()
        assert any("validation_base_url" in w for w in warnings)

    def test_non_url_base_url_flagged(self):
        cfg = CareConfig(memory={"base_url": "not-a-url"})
        warnings = cfg.audit_fields()
        assert any("memory.base_url" in w and "http" in w for w in warnings)

    def test_jwt_in_mage_model_flagged(self):
        cfg = CareConfig(mage=MageConfig(api_key="k", model=_JWT))
        warnings = cfg.audit_fields()
        assert any("mage.model" in w for w in warnings)

    def test_none_mage_url_is_ignored(self):
        # mage.base_url defaults to None — must not warn.
        cfg = CareConfig(mage=MageConfig(api_key="k"))
        assert all("mage.base_url" not in w for w in cfg.audit_fields())
