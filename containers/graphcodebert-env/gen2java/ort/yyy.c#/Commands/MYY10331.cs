namespace GEN.ORT.YYY
{
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  //
  //                    Source Code Generated by
  //                           CA Gen 8.6
  //
  //    Copyright (c) 2024 CA Technologies. All rights reserved.
  //
  //    Name: MYY10331_TYPE_UPDATE             Date: 2024/01/09
  //    Target OS:   CLR                       Time: 13:41:42
  //    Target DBMS: ODBC/ADO.NET              User: AliAl
  //    Access Method: <NONE>         
  //
  //    Generation options:
  //    Debug trace option not selected
  //    Data modeling constraint enforcement not selected
  //    Optimized import view initialization not selected
  //    Enforce default values with DBMS not selected
  //    Init unspecified optional fields to NULL not selected
  //
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  // using Statements
  // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
  using System;
  using com.ca.gen.vwrt;
  using com.ca.gen.vwrt.types;
  using com.ca.gen.vwrt.vdf;
  using com.ca.gen.csu.exception;
  
  using com.ca.gen.abrt;
  using com.ca.gen.abrt.functions;
  using com.ca.gen.abrt.cascade;
  using com.ca.gen.abrt.manager;
  using com.ca.gen.abrt.trace;
  using com.ca.gen.exits.common;
  using com.ca.gen.odc;
  using System.Data;
  using System.Collections;
  
  public class MYY10331 : ABBase
  {
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // IMPORT VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    MYY10331_IA WIa;
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // EXPORT VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    MYY10331_OA WOa;
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // LOCAL VIEW CLASS VARIABLE
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    MYY10331_LA WLa;
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // ACTION BLOCK IMPORT/EXPORT VIEWS CLASS VARIABLES
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    GEN.ORT.YYY.CYYY0331_IA Cyyy0331Ia;
    GEN.ORT.YYY.CYYY0331_OA Cyyy0331Oa;
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // MISC DECLARATIONS AND PROTOTYPES 
    //    FOLLOW AS NEEDED:             
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    bool func_0022020135_esc_flag;
    //       +->   MYY10331_TYPE_UPDATE              01/09/2024  13:41
    //       !       IMPORTS:
    //       !         Work View imp_reference iyy1_server_data (Transient,
    //       !                     Mandatory, Import only)
    //       !           userid
    //       !           reference_id
    //       !         Entity View imp iyy1_type (Transient, Mandatory, Import
    //       !                     only)
    //       !           tinstance_id
    //       !           treference_id
    //       !           tkey_attr_text
    //       !           tsearch_attr_text
    //       !           tother_attr_text
    //       !           tother_attr_date
    //       !           tother_attr_time
    //       !           tother_attr_amount
    //       !       EXPORTS:
    //       !         Entity View exp iyy1_type (Transient, Export only)
    //       !           treference_id
    //       !         Work View exp_error iyy1_component (Transient, Export
    //       !                     only)
    //       !           severity_code
    //       !           rollback_indicator
    //       !           origin_servid
    //       !           context_string
    //       !           return_code
    //       !           reason_code
    //       !           checksum
    //       !       LOCALS:
    //       !         Entity View loc_imp type
    //       !           tinstance_id
    //       !           treference_id
    //       !           tkey_attr_text
    //       !           tsearch_attr_text
    //       !           tother_attr_text
    //       !           tother_attr_date
    //       !           tother_attr_time
    //       !           tother_attr_amount
    //       !         Entity View loc_exp type
    //       !           treference_id
    //       !
    //       !     PROCEDURE STATEMENTS
    //       !
    //     1 !  NOTE: 
    //     1 !  See the description for the purpose
    //     1 !  
    //     2 !  NOTE: 
    //     2 !  RELEASE HISTORY
    //     2 !  01_00 23-02-1998 New release
    //     2 !  
    //     3 !  NOTE: 
    //     3 !  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //     3 !  !!!!!!!!!!!!
    //     3 !  SET <loc imp*> TO <imp*>
    //     3 !  
    //     4 !  SET loc_imp type tinstance_id TO imp iyy1_type tinstance_id 
    //     5 !  SET loc_imp type treference_id TO imp iyy1_type treference_id 
    //     6 !  SET loc_imp type tkey_attr_text TO imp iyy1_type
    //     6 !              tkey_attr_text 
    //     7 !  SET loc_imp type tsearch_attr_text TO imp iyy1_type
    //     7 !              tsearch_attr_text 
    //     8 !  SET loc_imp type tother_attr_text TO imp iyy1_type
    //     8 !              tother_attr_text 
    //     9 !  SET loc_imp type tother_attr_date TO imp iyy1_type
    //     9 !              tother_attr_date 
    //    10 !  SET loc_imp type tother_attr_time TO imp iyy1_type
    //    10 !              tother_attr_time 
    //    11 !  SET loc_imp type tother_attr_amount TO imp iyy1_type
    //    11 !              tother_attr_amount 
    //    12 !   
    //    13 !  NOTE: 
    //    13 !  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //    13 !  !!!!!!!!!!!!
    //    13 !  USE <implementation ab>
    //    13 !  
    //    14 !  USE cyyy0331_type_update
    //    14 !     WHICH IMPORTS: Entity View loc_imp type TO Entity View imp
    //    14 !              type
    //    14 !                    Work View imp_reference iyy1_server_data TO
    //    14 !              Work View imp_reference iyy1_server_data
    //    14 !     WHICH EXPORTS: Entity View loc_exp type FROM Entity View
    //    14 !              exp type
    //    14 !                    Work View exp_error iyy1_component FROM Work
    //    14 !              View exp_error iyy1_component
    //    15 !   
    //    16 !  NOTE: 
    //    16 !  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //    16 !  !!!!!!!!!!!!
    //    16 !  SET <exp*> TO <loc exp*>
    //    16 !  
    //    17 !  SET exp iyy1_type treference_id TO loc_exp type treference_id 
    //       +---
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    //  CONSTRUCTOR FOR THE CLASS       
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    public MYY10331(  )
    {
      IefCGenRlse = "CA Gen 8.6";
      IsCopyright = "Copyright (c) 2024 CA Technologies. All rights reserved.";
      IefCGenDate = "2024/01/09";
      IefCGenTime = "13:41:42";
      IefCGenEncy = "9.2.A6";
      IefCGenUserId = "AliAl";
      IefCGenModel = "N8I_ORT_YYY_0112_TEMPLATE";
      IefCGenSubset = "ALL";
      IefCGenName = "MYY10331_TYPE_UPDATE";
      NestingLevel = 0;
      ValChkDeadlockTimeout = false;
      ValChkDBError = false;
    }
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // ACTION BLOCK FUNCTION DECLARATIONS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    public void Execute( Object in_runtime_parm1, 
    	IRuntimePStepContext in_runtime_parm2, 
    	GlobData in_globdata, 
    	MYY10331_IA import_view, 
    	MYY10331_OA export_view )
    {
      IefRuntimeParm1 = in_runtime_parm1;
      IefRuntimeParm2 = in_runtime_parm2;
      Globdata = in_globdata;
      WIa = import_view;
      WOa = export_view;
      _Execute();
    }
    
    private void _Execute()
    {
      
      f_22020135_localAlloc( "MYY10331_TYPE_UPDATE" );
      if ( Globdata.GetErrorData().GetLastStatus() == ErrorData.LastStatusIefAllocationError )
      	return;
      
      ++(NestingLevel);
      try {
        f_22020135_init(  );
        f_22020135(  );
      } catch( Exception e ) {
        if ( ((Globdata.GetErrorData().GetStatus() == ErrorData.StatusNone) && (Globdata.GetErrorData().GetErrorEncountered() == 
          ErrorData.ErrorEncounteredNoErrorFound)) && (Globdata.GetErrorData().GetViewOverflow() == 
          ErrorData.ErrorEncounteredNoErrorFound) )
        {
          Globdata.GetErrorData().SetStatus( ErrorData.LastStatusFatalError );
          Globdata.GetErrorData().SetLastStatus( ErrorData.LastStatusUnexpectedExceptionError );
          Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredErrorFound );
          Globdata.GetErrorData(  ).SetErrorMessage( e );
        }
      }
      --(NestingLevel);
    }
    public void f_22020135(  )
    {
      func_0022020135_esc_flag = false;
      Globdata.GetStateData().SetCurrentABId( "0022020135" );
      Globdata.GetStateData().SetCurrentABName( "MYY10331_TYPE_UPDATE" );
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    See the description for the purpose                             
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    RELEASE HISTORY                                                 
      //    01_00 23-02-1998 New release                                    
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
      //    !!!!!!!!!!!!                                                    
      //    SET <loc imp*> TO <imp*>                                        
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      Globdata.GetStateData().SetLastStatementNumber( "0000000004" );
      WLa.LocImpTypeTinstanceId = TimestampAttr.ValueOf(WIa.ImpIyy1TypeTinstanceId);
      Globdata.GetStateData().SetLastStatementNumber( "0000000005" );
      WLa.LocImpTypeTreferenceId = TimestampAttr.ValueOf(WIa.ImpIyy1TypeTreferenceId);
      Globdata.GetStateData().SetLastStatementNumber( "0000000006" );
      WLa.LocImpTypeTkeyAttrText = FixedStringAttr.ValueOf(WIa.ImpIyy1TypeTkeyAttrText, 4);
      Globdata.GetStateData().SetLastStatementNumber( "0000000007" );
      WLa.LocImpTypeTsearchAttrText = FixedStringAttr.ValueOf(WIa.ImpIyy1TypeTsearchAttrText, 20);
      Globdata.GetStateData().SetLastStatementNumber( "0000000008" );
      WLa.LocImpTypeTotherAttrText = FixedStringAttr.ValueOf(WIa.ImpIyy1TypeTotherAttrText, 2);
      Globdata.GetStateData().SetLastStatementNumber( "0000000009" );
      WLa.LocImpTypeTotherAttrDate = DateAttr.ValueOf(WIa.ImpIyy1TypeTotherAttrDate);
      Globdata.GetStateData().SetLastStatementNumber( "0000000010" );
      WLa.LocImpTypeTotherAttrTime = TimeAttr.ValueOf(WIa.ImpIyy1TypeTotherAttrTime);
      Globdata.GetStateData().SetLastStatementNumber( "0000000011" );
      WLa.LocImpTypeTotherAttrAmount = DecimalAttr.ValueOf(TIRBDTRU.TruncateToDecimal( WIa.ImpIyy1TypeTotherAttrAmount, 2));
      
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
      //    !!!!!!!!!!!!                                                    
      //    USE <implementation ab>                                         
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      Globdata.GetStateData().SetLastStatementNumber( "0000000014" );
      
      Cyyy0331Ia = (GEN.ORT.YYY.CYYY0331_IA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.CYYY0331).Assembly,
      	"GEN.ORT.YYY.CYYY0331_IA" ));
      Cyyy0331Oa = (GEN.ORT.YYY.CYYY0331_OA)(IefRuntimeParm2.GetInstance( typeof(GEN.ORT.YYY.CYYY0331).Assembly,
      	"GEN.ORT.YYY.CYYY0331_OA" ));
      Cyyy0331Ia.ImpReferenceIyy1ServerDataUserid = FixedStringAttr.ValueOf(WIa.ImpReferenceIyy1ServerDataUserid, 8);
      Cyyy0331Ia.ImpReferenceIyy1ServerDataReferenceId = TimestampAttr.ValueOf(WIa.ImpReferenceIyy1ServerDataReferenceId);
      Cyyy0331Ia.ImpTypeTinstanceId = TimestampAttr.ValueOf(WLa.LocImpTypeTinstanceId);
      Cyyy0331Ia.ImpTypeTreferenceId = TimestampAttr.ValueOf(WLa.LocImpTypeTreferenceId);
      Cyyy0331Ia.ImpTypeTkeyAttrText = FixedStringAttr.ValueOf(WLa.LocImpTypeTkeyAttrText, 4);
      Cyyy0331Ia.ImpTypeTsearchAttrText = FixedStringAttr.ValueOf(WLa.LocImpTypeTsearchAttrText, 20);
      Cyyy0331Ia.ImpTypeTotherAttrText = FixedStringAttr.ValueOf(WLa.LocImpTypeTotherAttrText, 2);
      Cyyy0331Ia.ImpTypeTotherAttrDate = DateAttr.ValueOf(WLa.LocImpTypeTotherAttrDate);
      Cyyy0331Ia.ImpTypeTotherAttrTime = TimeAttr.ValueOf(WLa.LocImpTypeTotherAttrTime);
      Cyyy0331Ia.ImpTypeTotherAttrAmount = DecimalAttr.ValueOf(WLa.LocImpTypeTotherAttrAmount);
      Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredNoErrorFound );
      IefRuntimeParm2.UseActionBlock( typeof(GEN.ORT.YYY.CYYY0331).Assembly,
      	"GEN.ORT.YYY.CYYY0331",
      	"Execute",
      	Cyyy0331Ia,
      	Cyyy0331Oa );
      if ( ((Globdata.GetErrorData().GetStatus() != ErrorData.StatusNone) || (Globdata.GetErrorData().GetErrorEncountered() != 
        ErrorData.ErrorEncounteredNoErrorFound)) || (Globdata.GetErrorData().GetViewOverflow() != 
        ErrorData.ErrorEncounteredNoErrorFound) )
      {
        throw new ABException();
      }
      Globdata.GetStateData().SetCurrentABId( "0022020135" );
      Globdata.GetStateData().SetCurrentABName( "MYY10331_TYPE_UPDATE" );
      Globdata.GetStateData().SetLastStatementNumber( "0000000014" );
      WOa.ExpErrorIyy1ComponentSeverityCode = FixedStringAttr.ValueOf(Cyyy0331Oa.ExpErrorIyy1ComponentSeverityCode, 1);
      WOa.ExpErrorIyy1ComponentRollbackIndicator = FixedStringAttr.ValueOf(Cyyy0331Oa.ExpErrorIyy1ComponentRollbackIndicator, 1);
      WOa.ExpErrorIyy1ComponentOriginServid = DoubleAttr.ValueOf(Cyyy0331Oa.ExpErrorIyy1ComponentOriginServid);
      WOa.ExpErrorIyy1ComponentContextString = StringAttr.ValueOf(Cyyy0331Oa.ExpErrorIyy1ComponentContextString, 512);
      WOa.ExpErrorIyy1ComponentReturnCode = IntAttr.ValueOf(Cyyy0331Oa.ExpErrorIyy1ComponentReturnCode);
      WOa.ExpErrorIyy1ComponentReasonCode = IntAttr.ValueOf(Cyyy0331Oa.ExpErrorIyy1ComponentReasonCode);
      WOa.ExpErrorIyy1ComponentChecksum = FixedStringAttr.ValueOf(Cyyy0331Oa.ExpErrorIyy1ComponentChecksum, 15);
      WLa.LocExpTypeTreferenceId = TimestampAttr.ValueOf(Cyyy0331Oa.ExpTypeTreferenceId);
      Cyyy0331Ia.FreeInstance(  );
      Cyyy0331Ia = null;
      Cyyy0331Oa.FreeInstance(  );
      Cyyy0331Oa = null;
      
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      //    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
      //    !!!!!!!!!!!!                                                    
      //    SET <exp*> TO <loc exp*>                                        
      //                                                                    
      // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
      Globdata.GetStateData().SetLastStatementNumber( "0000000017" );
      WOa.ExpIyy1TypeTreferenceId = TimestampAttr.ValueOf(WLa.LocExpTypeTreferenceId);
      return;
    }
    
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // SUBORDINATE FUNCTIONS
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    // INITIALIZATION UTILITY FUNCTIONS 
    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    
    public void f_22020135_localAlloc( String abname )
    {
      // Request localview allocation 
      WLa = (GEN.ORT.YYY.MYY10331_LA)(IefRuntimeParm2.GetInstance( GetType().Assembly,
      	"GEN.ORT.YYY.MYY10331_LA" ));
      if ( WLa == null )
      {
        Globdata.GetStateData().SetCurrentABId( "0022020135" );
        Globdata.GetStateData().SetCurrentABName( abname );
        Globdata.GetErrorData().SetErrorEncountered( ErrorData.ErrorEncounteredErrorFound );
        Globdata.GetErrorData().SetLastStatus( ErrorData.LastStatusIefAllocationError );
      }
    }
    
    public void f_22020135_init(  )
    {
      if ( NestingLevel < 2 )
      {
        WLa.Reset();
      }
      WOa.ExpIyy1TypeTreferenceId = "00000000000000000000";
      WOa.ExpErrorIyy1ComponentSeverityCode = " ";
      WOa.ExpErrorIyy1ComponentRollbackIndicator = " ";
      WOa.ExpErrorIyy1ComponentOriginServid = 0.0;
      WOa.ExpErrorIyy1ComponentContextString = "";
      WOa.ExpErrorIyy1ComponentReturnCode = 0;
      WOa.ExpErrorIyy1ComponentReasonCode = 0;
      WOa.ExpErrorIyy1ComponentChecksum = "               ";
    }
  }// end class
  
}

